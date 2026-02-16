import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str #模型路径：指定你要加载的模型在哪里。可以是 HuggingFace 的 Hub ID（如 Qwen/Qwen3-0.6B），也可以是本地文件夹的绝对路径
    max_num_batched_tokens: int = 16384 #限制 GPU 在一次前向传播（Step）中最多能处理多少个 Token。
    #主要限制 Prefill（预填充）阶段。如果来了10个请求，每个长2000Token，总共20000Token，超过了16384，那么系统就不能一次性把这10个请求都放进去跑，可能只能先跑8个。这为了防止显存爆掉（OOM）
    max_num_seqs: int = 512 #最大并发序列数：限制系统同时能处理多少个请求（Request）
    max_model_len: int = 4096 #模型支持的最大上下文长度。作用: 限制 Prompt + Output 的总长度。如果用户发来的 Prompt 太长，或者生成的太长超过这个数，会被截断或报错
    gpu_memory_utilization: float = 0.9 #GPU 显存占用比例，Nano-vLLM 会预先占用 GPU 显存来存放 KV Cache，剩下的 10% 留给 PyTorch 的临时激活值（Activation）开销，防止计算时 OOM
    tensor_parallel_size: int = 1 #张量并行度 (TP Size) 大于1表示用n张卡跑一个模型，模型的每一层（Linear 层）会被切分到n张卡上计算，中间需要 AllReduce 通信
    enforce_eager: bool = False #强制使用 Eager 模式，False (默认)则尝试使用CUDA Graph技术来加速 Decode 阶段（录制计算图，减少 CPU 启动 GPU Kernel 的开销）
    hf_config: AutoConfig | None = None #HuggingFace 配置对象，初始化时它是 None，但在 __post_init__ 里会被填充。它包含了模型具体的架构细节（如有多少层、隐藏层维度是多少、头数是多少），供 ModelRunner 构建模型网络使用
    eos: int = -1 #结束符 Token ID (End of Sentence)
    kvcache_block_size: int = 256 #KV Cache 的块大小 (Block Size) PagedAttention 的核心参数。类似于操作系统的“页大小”（Page Size）。
        # 256 表示一个显存块（Block）能存 256 个 Token 的 KV 数据。块越大，显存碎片越少，但浪费可能越多（Tail Latency）；块越小，调度越灵活，但管理开销越大。
    num_kvcache_blocks: int = -1 #KV Cache 的块总数。这是一个计算结果。ModelRunner 启动时，会根据显存大小和 gpu_memory_utilization 算出到底能分出多少个块，填入这个变量

    def __post_init__(self): #Config(...) 初始化完成后，Python 会自动调用这个方法。通常用来做参数校验和进一步的初始化
        assert os.path.isdir(self.model) #校验模型路径是否存在
        assert self.kvcache_block_size % 256 == 0 #强制块大小必须是 256 的倍数（为了对齐硬件或内核优化的要求）
        assert 1 <= self.tensor_parallel_size <= 8 #限制并行度在 1 到 8 之间（通常单机最多 8 卡）
        self.hf_config = AutoConfig.from_pretrained(self.model) #调用 HuggingFace 的接口，读取模型文件夹下的 config.json 文件，解析出模型结构参数，存入 self.hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) 
        #这是一个安全逻辑，用户可能在配置里写 max_model_len = 100000。但如果模型本身的 config.json 里写着只支持 2048（max_position_embeddings），则这里取最小值
        assert self.max_num_batched_tokens >= self.max_model_len #校验: 确保吞吐量限制至少能容纳一个最长的请求。如果一次能处理的 Token 还没一个句子的最大长度多，那这个系统就跑不起来了
