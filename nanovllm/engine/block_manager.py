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
        这样就不用重新计算，直接复用显存，不仅省空间，还能跳过计算（Prefill 变快）。"""
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
        self.hash_to_block_id: dict[int, int] = dict() #哈希索引表 Key: 块内容的哈希值 (int) -> Value: 物理块 ID (int)
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #空闲块池 使用 deque (双端队列) 存储所有没被使用的块 ID。
        self.used_block_ids: set[int] = set() #已用块集合 (The Occupied List) 使用 set (集合) 记录正在使用的块 ID。作用：O(1)快速查询某个块是否被占用，防止重复分配或错误回收。

    @classmethod 
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): #生成指纹(哈希密码)
        h = xxhash.xxh64() # 1. 创建哈希生成器 这里用了 xxhash (xxh64)，这是一种非加密型哈希算法。特点：速度极快！比 MD5 或 SHA 快得多，非常适合这种高频调用的场景。
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
