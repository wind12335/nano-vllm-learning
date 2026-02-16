from collections import deque

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
        self.waiting: deque[Sequence] = deque()  # 等待队列：新来的请求，还没开始跑
        self.running: deque[Sequence] = deque()  # 运行队列：正在 Decode 生成中的请求

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: #这个函数在 LLMEngine.step() 中被调用。
        # 它的逻辑非常清晰：优先做 Prefill，没 Prefill 再做 Decode。这也是 Nano-vLLM 与标准 vLLM 的一个简化点（标准 vLLM 支持 Prefill 和 Decode 混合跑）
        # --- Prefill 逻辑 ---
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # 只要 Waiting 队列里还有人，且没超过最大并发限制
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] #以此看看队首的请求
            
            # 检查 1: Token 数量是否超标？ 如果加上这个请求，总 Token 数超过了 max_num_batched_tokens，这轮就不带它玩了
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or \
               not self.block_manager.can_allocate(seq): # 检查 2: 显存够不够？
                break
            
            # 决定调度这个请求
            num_seqs += 1
            self.block_manager.allocate(seq) # 分配显存 Block
            
            # 计算这次 Prefill 实际要算的 Token 数 (len - cached)
            # 这里考虑了 Prefix Caching：如果前缀已经缓存过，就不需要重新计算了
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            
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
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 取出一个请求
            
            # 检查显存: "我要生成下一个 Token 了，显存够给我追加一个 Slot 吗？" 如果 Block 满了需要新开一个 Block，而显存池空了，就会进入 while 循环
            while not self.block_manager.can_append(seq):
                # 显存不够了！触发抢占机制 (Swap out)
                if self.running:
                    # 牺牲策略：把 Running 队列队尾的请求踢出去 (最晚加入的)
                    self.preempt(self.running.pop())
                else:
                    # 如果只剩我自己了还是不够，那我也只能被踢出去
                    self.preempt(seq)
                    break
            else:
                # 显存足够 (或通过抢占腾出了空间)
                num_seqs += 1
                self.block_manager.may_append(seq) # 尝试追加物理块 (如果当前块满了)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs
        # 把这一轮选中的请求放回 running 队列头部，保持顺序
        self.running.extendleft(reversed(scheduled_seqs))
        # 返回结果，is_prefill = False
        return scheduled_seqs, False

    def preempt(self, seq: Sequence): #抢占机制
        seq.status = SequenceStatus.WAITING # 降级为 WAITING
        self.block_manager.deallocate(seq)  # ★ 关键：释放它占用的所有显存块
        self.waiting.appendleft(seq)        # 插队到 Waiting 队列的最前面 (高优先级)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 把新生成的 Token 加进 Sequence
            
            # 判断是否结束： 1. 生成了 EOS 结束符 (且没开启 ignore_eos) 2. 长度达到了 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                
                seq.status = SequenceStatus.FINISHED # 标记完成
                self.block_manager.deallocate(seq)   # ★ 立即释放显存！
                self.running.remove(seq)             # 从运行队列移除
