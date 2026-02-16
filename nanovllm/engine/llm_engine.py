import atexit #atexit: 用于注册程序退出时需要执行的清理函数（比如关闭显存共享
from dataclasses import fields
from time import perf_counter #高精度计时器，用于计算吞吐量（Tokens/s）
from tqdm.auto import tqdm #进度条库，用于在 generate 时显示进度
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config #配置
from nanovllm.sampling_params import SamplingParams #采样参数
from nanovllm.engine.sequence import Sequence #序列对象
from nanovllm.engine.scheduler import Scheduler #调度器
from nanovllm.engine.model_runner import ModelRunner #模型执行器                     


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 1. 配置参数过滤与初始化
        config_fields = {field.name for field in fields(Config)} #得到的{'max_num_seqs', 'gpu_memory_utilization', 'tensor_parallel_size', ...}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} #清洗参数，筛出kwargs中config_fields出现的变量，输出键值对
        # **config_kwargs: 这是 Python 的字典解包 (Dictionary Unpacking) 语法。它把字典里的键值对转换成关键字参数传给函数。相当于：Config(model, max_num_seqs=..., tensor_parallel_size=..., ...)。
        config = Config(model, **config_kwargs)
        # 2. 多进程初始化 (用于张量并行)
        self.ps = []      # 存储子进程对象
        self.events = []  # 存储进程间同步的 Event 对象
        ctx = mp.get_context("spawn") # 使用 'spawn' 模式启动进程 (CUDA 环境下必须用 spawn 而非 fork)

        # 循环启动 Rank 1 到 Rank N-1 的子进程
        # Rank 0 (主进程) 负责协调和一部分计算，其他 Rank 只负责计算
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # target=ModelRunner: 子进程直接运行 ModelRunner 类
            # args=(config, i, event): 传入配置、当前进程的 Rank (编号)、同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) #创建了一个新的操作系统进程对象（但还没开始运行）启动之后立刻去执行 ModelRunner 这个类
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 3. 初始化主进程 (Rank 0) 的组件
        # 主进程也需要一个 ModelRunner 来参与计算和管理
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 4. 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id # 将 EOS token ID 注入配置中，供后续判断生成结束使用
        
        # 5. 初始化调度器
        self.scheduler = Scheduler(config)
        
        # 6. 注册退出清理函数
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit") # 发送指令给 ModelRunner，让它通知所有子进程退出循环
        del self.model_runner          # 删除主进程的 runner，触发其析构函数中的资源释放
        for p in self.ps:
            p.join()                   # 等待所有子进程安全结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果输入是文本，先进行分词 (Encode) 转为 token ids
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) #括号里的prompt是用户输入的提示词，输出的prompt可以是字符串或已经分词好的 ID 列表
        # 创建一个 Sequence 对象  Sequence 封装了 prompt tokens 和采样参数，是调度器管理的基本单位
        seq = Sequence(prompt, sampling_params) #采样参数（温度、最大长度等）
        # 将请求加入调度器的等待队列 (Waiting Queue)
        self.scheduler.add(seq)
     
    def step(self):
        # 1. 调度 (Schedule)  询问调度器：下一轮我该跑哪些 Sequence？是做 Prefill 还是 Decode？
        # seqs: 本次被选中执行的 Sequence 列表  is_prefill: 布尔值，True 表示这批是新来的请求（预填充阶段），False 表示是正在生成的请求（解码阶段）
        seqs, is_prefill = self.scheduler.schedule()
        
        # 2. 执行模型 (Run) 调用 ModelRunner 在 GPU 上执行模型 返回 token_ids: 每个 Sequence 生成的下一个 Token 的 ID
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 3. 后处理 (Postprocess) 将生成的 Token 追加到 Sequence 中 检查是否遇到 EOS 或达到最大长度，如果是，将 Sequence 标记为 FINISHED 并释放资源
        self.scheduler.postprocess(seqs, token_ids)
        
        # 4. 收集输出 筛选出刚刚完成 (FINISHED) 的 Sequence，提取其结果 seq.completion_token_ids 指的是模型新生成的那部分 Token 的 ID（不包含 Prompt 的 ID）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 5. 计算吞吐量统计数据
        # 这是一个小技巧 如果是 Prefill，处理的 Token 数是所有 Prompt 的长度之和，如果是 Decode，每个 Sequence 只生成 1 个 Token，所以是 -len(seqs)
        # (用负数是为了在 generate 函数里方便区分是 Prefill 还是 Decode 阶段来打印日志)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        # outputs是 Token IDs (整数列表)不是文字。 例如：[(0, [123, 456, 789]), (1, [321, 654])]。LLMEngine.generate() 函数的最后（llm_engine.py 第 78 行）才转为文字
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() # 检查调度器里是否还有东西 只要 Waiting 队列或 Running 队列不为空，就说明还没干完活

    def generate(
        self, prompts: list[str] | list[list[int]], #prompts：文本["你好", "你是谁"] 或 已分好词的Token ID：[[101, 234], [101, 567]]
        sampling_params: SamplingParams | list[SamplingParams], #可以是一个单独的 SamplingParams 对象，也可以是一个装满 SamplingParams 对象的列表
        # 想用同一套参数（比如温度都是 0.7）跑 100 个 Prompt则只需要传一个 SamplingParams 对象，若第1个想用温度 0.1（严谨），第2个想用温度 0.9（发散）就可以传一个包含两个不同参数对象的列表 [param1, param2
        use_tqdm: bool = True,
    ) -> list[str]:
        # 初始化进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) #tqdm 库创建的一个进度条对象
        
        # 参数对齐：如果只给了一个采样参数，就复制给所有 Prompt
        if not isinstance(sampling_params, list): # isinstance() 判断一个对象是否是一个已知的类型，这里判断sampling_params是否是list
            sampling_params = [sampling_params] * len(prompts) # 假设传进来的sampling_params是对象SP，prompts有3个。这行代码会生成一个新的列表：[SP, SP, SP]
            
        # 1. 将所有请求加入调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 2. 循环直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            
            # ★ 核心调用：执行一步推理
            output, num_tokens = self.step()
            
            # 更新进度条和吞吐量显示
            if use_tqdm:
                # 利用 num_tokens 的正负号判断是 Prefill 还是 Decode
                if num_tokens > 0: # Prefill 阶段
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:              # Decode 阶段
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集这一步完成的请求的结果
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1) # 每完成一个请求，进度条 +1
                    
        # 3. 整理结果并解码
        # 按 seq_id 排序确保输出顺序与输入顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 将 Token IDs 解码回文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        return outputs


"""让我们再看一遍 step 函数的完整剧本：

调度 (scheduler.schedule)：
调度器："我有 10 个请求在排队，显存够用，这轮先处理这 10 个请求的 Prefill 吧。" -> 返回 is_prefill=True。
或者："刚才那 10 个请求 Prefill 完了，显存还有剩，这轮让它们每人生成 1 个新 Token (Decode)。" -> 返回 is_prefill=False。

执行 (model_runner.run)：
ModelRunner：收到指令。如果是 Prefill，全速并行计算；如果是 Decode，利用 PagedAttention 快速读取 KV Cache 并计算。
返回：所有请求这轮新生成的 Token ID（比如每个请求返回 1 个 ID）。

后处理 (scheduler.postprocess)：
调度器：把新 ID 加到 Sequence 对象里。检查："哟，请求 A 生成了句号（EOS），它完事了，标记为 FINISHED，回收它的显存 Block。"

收集 (outputs)：把刚才标记为 FINISHED 的请求 A 的结果打包。请求 B 还没完，就不放进 outputs。

统计 (num_tokens)：计算这轮 GPU 到底吞吐了多少 Token，用于显示 "速度: 1500 tokens/s"。
"""