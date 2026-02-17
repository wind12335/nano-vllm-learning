from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

"""这个代码实现了pagedAttention"""

class Block:

    def __init__(self, block_id):
        self.block_id = block_id #显存里的物理块 ID(唯一)ModelRunner 和 GPU Kernel 中，代码会根据这个 ID 算出显存的物理地址偏移量。例如：offset = block_id * block_size * hidden_dim
        self.ref_count = 0 #多少个逻辑请求（Sequence）正在引用（使用）这个物理块
        """hash原理
        当一个 Block 填满（比如存了 16 个 Token [101, 20, ..., 5]）时，我们会对这串数字算一个 Hash 值（比如 hash = 123456）。
        下次再来一个请求，如果它算出来的 Hash 也是 123456，说明它的内容和之前一模一样。BlockManager 就可以直接查表：if 123456 in cached_blocks: return block_id。
        这样就不用重新计算，直接复用显存，不仅省空间，还能跳过计算（Prefill 变快）。
        """
        self.hash = -1 #这个块里存储的 Token 内容的指纹。 作用：用于 Prefix Caching（前缀缓存） 的快速查找
        self.token_ids = [] #这个物理块里实际存放的 Token ID 列表

    def update(self, hash: int, token_ids: list[int]): # 作用：为“前缀缓存”（Prefix Caching）做准备。告知系统“这个块里装的是 A, B, C 这些数据，指纹(hash)是 12345”。
                                                       # 以后如果再有人需要 A, B, C，BlockManager 拿着指纹一比对，就能找到这个块
        self.hash = hash #计算好的哈希值（指纹）存入 Block 对象
        self.token_ids = token_ids #这个块里实际包含的 Token 列表存下来

    def reset(self): #通常在 从空闲池（Free Pool）里拿出一个旧块给新请求用 时调用
        self.ref_count = 1 #调用 reset 意味着这个块刚刚被分配给了一个新的请求（Sequence）。所以引用计数初始化为 1
        self.hash = -1 #哈希值重置为无效值（-1） 旧的指纹作废
        self.token_ids = [] #清空旧的 Token 列表


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size 
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 一次性创建所有 Block 对象。比如 num_blocks=100，这里就生成 100 个 Block 实例。
        self.hash_to_block_id: dict[int, int] = dict() #哈希索引表 Key: 块内容的哈希值 (int) -> Value: 物理块 ID (int) dict()就是字典
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #空闲块池 使用 deque (双端队列) 存储所有没被使用的块 ID。
        self.used_block_ids: set[int] = set() #已用块集合 (The Occupied List) 使用 set (集合) 记录正在使用的块 ID。作用：O(1)快速查询某个块是否被占用，防止重复分配或错误回收。

    @classmethod 
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): #生成指纹(哈希密码)
        h = xxhash.xxh64() # 1. 创建哈希生成器 这里用了 xxhash (xxh64)，这是一种非加密型哈希算法。特点：速度极快！比 MD5 或 SHA 快得多，非常适合这种高频调用的场景。

        # 2. 链接前缀 (Chaining) 如果这个块不是第一个块 (prefix != -1)，我们要把“上一个块的哈希值”也加进来一起算。
        # 为什么？因为 "A -> B" 和 "C -> B" 是不同的语义。 哪怕 B 的内容一样，如果前文不同，这个块也不能混用。
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))# 把前缀整数转成字节
        # 3. 加上当前内容 把当前的 token_ids 列表转成 numpy 数组再转成字节流，喂给哈希函数。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest() # 4. 生成最终指纹

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]# 1. 拿到对应的 Block 对象
        assert block.ref_count == 0  # 2. 安全检查：确保这个块真的是空的
        block.reset() # 把 ref_count 设为 1，清空旧数据。这是BLOCK里的的 reset()函数。
        # 从“空闲池”移除，加入“已用集合”。
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0 #检查引用是否为0
        # 从“已用集合”移除，加入“空闲池”。
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        # seq.num_blocks: 这个请求总共需要多少个房间？
        # len(self.free_block_ids): 我们可以用的空房间还有多少？
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table   # 确保 Sequence 是干净的，没分配过
        h = -1                       # 初始哈希值 (链头)
        cache_miss = False           # 缓存未命中标志：一旦断了，后面全都要新开
        for i in range(seq.num_blocks): #遍历每一个逻辑块
            token_ids = seq.block(i)
            # 只有满的块才配有哈希值；不满的块（通常是最后一个）不缓存，h 传进去作为 prefix，实现了“链式哈希”：当前块的指纹 = Hash(前一块指纹 + 当前块内容)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 去“老客户名册”里查查这个指纹有没有对应的物理房间
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: #两个条件算“未命中”：1.block_id==-1: 名册里没查到  2.内容不匹配:查到了但内容不对(哈希冲突防御)
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else: #缓存命中 (Cache Hit!) —— 省显存时刻
                seq.num_cached_tokens += self.block_size # 记录已缓存的token大小
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1 ## 引用计数 +1，和别人共用
                else: # 情况2: 这个物理块虽然在名册里，但目前没人用 (可能是刚释放但还没被清理)
                    block = self._allocate_block(block_id)
            if h != -1: 
                #如果生成了有效的哈希，就把这个块的信息更新进名册，方便后来人复用
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id 
            seq.block_table.append(block_id) #★ 最重要的一步：把物理房间号填入 Sequence 的块表

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): #倒叙查块表
            block = self.blocks[block_id] #拿到对应块
            block.ref_count -= 1  #引用-1
            if block.ref_count == 0:  #引用=0则直接释放
                self._deallocate_block(block_id)
        #3. 清理 Sequence 自身的数据
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool: #检查空闲块的数量 (len(free_block_ids)) 是否足够。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) #(len(seq) % self.block_size == 1)这里计算是否需要开新块

    #这个函数在每次生成一个 Token 后调用。它处理显存块的状态流转
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # --- 分支 1: 刚跨入新块 (New Block Needed) ---
        if len(seq) % self.block_size == 1:
            # 断言：上一个块肯定已经封存归档了 (hash != -1)
            assert last_block.hash != -1
            # 动作：开新房
            block_id = self.free_block_ids[0]     # 拿钥匙
            self._allocate_block(block_id)        # 登记
            block_table.append(block_id)          # 给客人房卡
        
        # --- 分支 2: 当前块刚填满 (Block Full & Finalize) ---
        elif len(seq) % self.block_size == 0:
            # 断言：这个块还在用，还没归档 (hash == -1)
            assert last_block.hash == -1
            
            # 动作：归档封存 (生成指纹，存入缓存)
            # 1. 拿到这个块里所有的 Token
            token_ids = seq.block(seq.num_blocks-1)
            # 2. 拿到前一个块的哈希 (为了做链式哈希)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 3. 计算当前块的哈希
            h = self.compute_hash(token_ids, prefix)
            # 4. 更新块信息 (存入 hash 和 token_ids)
            last_block.update(h, token_ids)
            # 5. 记入名册 (hash -> block_id)  这样以后别的请求如果生成了一样的内容，就可以复用这个块了！
            self.hash_to_block_id[h] = last_block.block_id
            
        # --- 分支 3: 还在当前块中间 (Normal Append) ---
        else:
            # 断言：还没填满，所以不能有哈希
            assert last_block.hash == -1
            # 什么都不用做！显存是预先分配好的，往里填数据是 Kernel 的事，
            # BlockManager 只管“块”级别的申请和释放。
