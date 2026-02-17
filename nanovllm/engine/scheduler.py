from collections import deque #Double-Ended Queue（双端队列）允许高效地从两头（左边或右边）添加或删除元素

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager #调度器本身不管理具体的显存块 ID，它只负责决策。具体的“切块”和“记账”工作交给 BlockManager

""" 这个类里实现了连续批处理 Continuous Batching"""
class Scheduler:

    def __init__(self, config: Config):
        # 限制条件
        self.max_num_seqs = config.max_num_seqs                # 比如最多同时处理 512 个请求
        self.max_num_batched_tokens = config.max_num_batched_tokens # 比如一次最多处理 16384 个 Token (针对 Prefill)
        self.eos = config.eos
        
        # 显存管家：Scheduler 必须时刻询问它“显存够不够？”
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 两个核心队列 (双端队列 deque) 使用双端队列是为了实现 FCFS（先来先服务）。新请求从右边进 (append)，调度时从左边取 (popleft)
        self.waiting: deque[Sequence] = deque()  # 等待队列：新来的请求，还没开始跑 这个队列里装的所有元素，都必须是 Sequence 类的实例对象
        self.running: deque[Sequence] = deque()  # 运行队列：正在 Decode 生成中的请求

    def is_finished(self):
        #not self.waiting: 意思是“等待队列是空的吗？”   not self.running: 意思是“运行队列是空的吗？
        return not self.waiting and not self.running #返回true或false

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: #这个函数在 LLMEngine.step() 中被调用。
        # 它的逻辑非常清晰：优先做 Prefill，没 Prefill 再做 Decode。这也是 Nano-vLLM 与标准 vLLM 的一个简化点（标准 vLLM 支持 Prefill 和 Decode 混合跑）
        # --- Prefill 逻辑 ---
        scheduled_seqs = [] #一个序列（Sequence）= 一个用户发来的请求。包含了用户输入的Prompt（一串 Token）以及后续生成出来的所有 Token。可以理解为“一个正在进行的对话任务”
        num_seqs = 0 #指“序列的数量”（Batch Size）。即 scheduled_seqs 这个列表里目前攒了多少个请求
        num_batched_tokens = 0 #指的是本轮所有被选中的序列，加起来总共有多少个 Token 需要计算，防止 GPU 显存爆掉（OOM）
        # 举例（Prefill 阶段），假设来了 2 个请求：请求A：Prompt 长度 10,000 Token。请求B：Prompt 长度 8,000 Token。虽然 num_seqs 只有 2（没有超过最大并发数 16）。
        # 但是总 Token 数 num_batched_tokens = 18,000。如果你的显存只够一次算 16,000 个 Token，那这两个请求就不能一起跑
        
        # 只要 Waiting 队列里还有人，且没超过最大并发限制
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] #以此看看队首的请求
            
            # 检查 1: Token 数量是否超标？ 如果加上这个请求，总 Token 数超过了 max_num_batched_tokens，这轮就不带它玩了
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or  not self.block_manager.can_allocate(seq): # 检查 2: 显存够不够？
                #can_allocate(seq)检查len(seq)的请求的 KV Cache 需要多少个 Block？如果显存池里剩下的空闲 Block 不够了，为了防止显存溢出（OOM），系统必须拒绝这个请求上车
                break #len(seq)指的是这条序列（请求）目前的总 Token 数
            
            # 决定调度这个请求
            num_seqs += 1
            self.block_manager.allocate(seq) # 分配显存 Block
            
            # 计算这次 Prefill 实际要算的 Token 数 (len - cached) 这里考虑了 Prefix Caching：如果前缀已经缓存过，就不需要重新计算了
            num_batched_tokens += len(seq) - seq.num_cached_tokens #token总长 - 已经存在显存里的kv = 真正要处理的token数(Embedding、QKV计算、attention、layernorm、MLP全来一遍)
            
            seq.status = SequenceStatus.RUNNING # 状态变更为 RUNNING
            self.waiting.popleft()              # 移出等待队列
            self.running.append(seq)            # 加入运行队列
            scheduled_seqs.append(seq)
            
        # ★ 如果这一轮选中了任何 Waiting 请求，直接返回！ is_prefill = True
        if scheduled_seqs:
            return scheduled_seqs, True
        #逻辑：贪心地从 waiting 队列头部取请求。
        # can_allocate(seq): 这是 PagedAttention 的精髓。在让请求上车前，必须先确保存放它的 KV Cache 所需的物理块（Block）是足够的。如果显存不够，就停止调度，哪怕并发数还没满

        # --- Decode 逻辑 ---
        # 遍历所有正在 Running 的请求 
        while self.running and num_seqs < self.max_num_seqs: #如果running队列里有seq且处理的seq数不大于限制数
            seq = self.running.popleft() # 取出一个decode请求
            
            # 检查显存: "我要生成下一个 Token 了，显存够给我追加一个 Slot 吗？" 如果 Block 满了需要新开一个 Block，而显存池空了，就会进入 while 循环
            while not self.block_manager.can_append(seq): #can_append(seq) 会判断“如果给seq再追加一个 Token，是不是需要开辟新块？如果需要，显存池里剩下的空闲块（Free Blocks）够不够？
                # 显存不够了！触发抢占机制 (Swap out)
                if self.running:
                    # 牺牲策略：把 Running 队列队尾的请求踢出去 (最晚加入的)
                    self.preempt(self.running.pop()) #self.running.pop() 是从队尾拿请求(队尾都是新的，队头是老的)
                else:
                    # 如果只剩我自己了还是不够，那我也只能被踢出去
                    self.preempt(seq)
                    break
            else:
                # 显存足够 (或通过抢占腾出了空间)
                num_seqs += 1
                self.block_manager.may_append(seq) # 尝试追加物理块 (如果当前块满了)注意这里是正式分配物理块
                scheduled_seqs.append(seq) #加入调度序列
        
        assert scheduled_seqs
        # 把这一轮选中的请求放回 running 队列头部，保持顺序(之前running.popleft()取出去了，现在在放回来)
        self.running.extendleft(reversed(scheduled_seqs)) #extendleft从左侧批量扩展，reversed反转
        # extendleft 的工作原理是：把列表里的元素，一个接一个地从左边塞进去，假设不反转，直接插 [A, B]：
        # 先把 A 从左边塞进去 -> 队列变成 [A, C, D] 再把 B 从左边塞进去 -> 队列变成 [B, A, C, D]，这样顺序变了！ 原本是 [A, B...]，现在变成了 [B, A...]。
        # 如果每次都这样，队列顺序就乱套了。所以，我们需要“负负得正”：
        # 先反转 (reversed)：把 [A, B] 变成 [B, A],再左插 (extendleft)：先把 B 从左边塞进去 -> 队列变成 [B, C, D],再把 A 从左边塞进去 -> 队列变成 [A, B, C, D]
        # 完美！ 队列恢复成了 [A, B, C, D]，和最开始一模一样
       
        return scheduled_seqs, False # 返回结果，is_prefill = False

    def preempt(self, seq: Sequence): #抢占机制
        seq.status = SequenceStatus.WAITING # 降级为 WAITING
        self.block_manager.deallocate(seq)  # ★ 关键：释放它占用的所有显存块
        self.waiting.appendleft(seq)        # 插队到 Waiting 队列的最前面 (高优先级) appendleft是一个人插队，而extendleft是一组人插队(这样必须颠倒顺序)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids): #zip(seqs, token_ids)形成了元组
            seq.append_token(token_id) # 把新生成的 Token 加进 Sequence 直接修改了传入的 seq 对象内部的数据，调度器改了之后，外面的 LLMEngine 看到的 seq 也就变
            
            # 如果1.用户允许正常结束 (not ignore_eos)，并且模型真的输出了结束符 (token_id == eos)，或 2.长度达到了 max_tokens 那么这个请求就结束了
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                 #num_completion_tokens: 增量部分，即模型一共吐了多少个字
                seq.status = SequenceStatus.FINISHED # 标记完成
                self.block_manager.deallocate(seq)   # ★ 立即释放显存！
                self.running.remove(seq)             # 从运行队列移除
