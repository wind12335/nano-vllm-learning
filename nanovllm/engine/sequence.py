from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum): 
    # auto():自动分配数值 如果不使用 auto()，你必须手动给每个状态指定一个具体的值，使用 auto()会自动填上这些值（通常是从 1 开始递增的整数），就不需要关心具体的数字是几了
    WAITING = auto()   # 自动变成 1
    RUNNING = auto()   # 自动变成 2
    FINISHED = auto()  # 自动变成 3


class Sequence:
    block_size = 256  # 默认块大小，用于计算需要多少个 Block
    counter = count() # 全局计数器，用于生成唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 数据存储
        self.token_ids = copy(token_ids)     # 初始化时，此时只有 Prompt Tokens
        self.last_token = token_ids[-1]      # 记录最后一个字，用于生成下一个

        # 计数器
        self.num_tokens = len(self.token_ids)       # 总长度 (Prompt + Output)
        self.num_prompt_tokens = len(token_ids)     # 用户发来的Prompt 长度 (固定)
        self.num_cached_tokens = 0                  # 核心：已经缓存的 Token 数

        # ★★★ 最核心的数据结构：块表(相当于页表)
        self.block_table = [] 
        # 这是一个整数列表，例如 [102, 58, 7]。含义：它记录了逻辑块到物理块的映射。如：逻辑块0 -> 物理显存块102    逻辑块1 -> 物理显存块58    逻辑块2 -> 物理显存块7
        # 作用：PagedAttention 算法就是靠查这张表，知道去显存的哪个位置读取 KV Cache

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens #llm_engine里的len(seq) 实际上就是去调用了 Sequence 类里的 __len__ 方法，拿回了 self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key] #定义当使用中括号[]对对象进行索引或切片时应该返回什么。如果不写它：调用 seq[0]或seq[1:5]会报错，提示对象不可下标（not subscriptable）。
        # 写了它之后：它让 Sequence 对象伪装成了一个列表。你可以直接把它当成存放 Token ID 的列表来用

    """什么是 @property？
    这个装饰器的作用是“把方法伪装成属性（变量）”。 普通方法：调用时需要加括号。比如 seq.get_status()。
    Property 方法：调用时不需要加括号，就像访问一个普通变量一样。写代码时只需写 seq.is_finished（看起来像读取一个变量）。
    Python 后台会自动执行 def is_finished(self): ... 这段代码，并把 return 的结果给你 写 if seq.is_finished: 比写 if seq.is_finished(): 读起来更像自然语言，代码更简洁
    """
    @property
    def is_finished(self):  #是否处理完毕   
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):  #获取decode阶段新生成的token数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): #获取初始token id(用户的prompt的token id)
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): #获取decode阶段新生成的token id
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): #获取有多少个完整的显存块是可以复用的 已缓存的Token总数 除以 块大小（向下取整）
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一块的token长度= 总长度 - (最后一块之前的完整块数 * 块大小)
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): #获取某个块的数据 (block)
        assert 0 <= i < self.num_blocks # 确保索引i在范围内
        # 切片操作：取出第 i 个逻辑块对应的所有 Token ID
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int): #添加token (decode阶段使用)
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    """
    什么是“钩子函数” (Hook Function)？在编程中，“钩子”就是“拦截点”。
    标准流程：通常，一个系统或库有一套标准的操作流程（比如：保存文件、发送网络请求、关闭窗口）。
    挂钩子：为了让开发者能干预这个流程，系统会在关键步骤预留一些“缺口”。如果你在这些缺口里填入了自己的函数，系统运行到这里时，就会“钩住”你的函数去执行。这就叫钩子函数。
    __getstate__ 和 __setstate__ 是pickle 模块的钩子函数，pickle 是 Python 用来把内存里的对象变成二进制流（保存到文件或网络传输）的工具
    """
    def __getstate__(self): #打包前的拦截
        # 触发时机：当使用 pickle.dump() 序列化一个对象时。 默认行为：如果没有定义这个函数，Python 会默认把对象的 self.__dict__（也就是对象里所有的属性字典）全部打包带走。
        # 自定义作用：“我要自己决定带什么走。” 可以选择性地丢弃一些没用的数据（比如临时的缓存、打开的文件句柄）,可以修改数据格式，让包变得更小。
        # 返回值：你返回什么，Python 就把什么存起来
        
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)
        #对象压缩成了一个包含5个元素的元组 (Tuple)，前四个元素是整数。如果是prefill阶段：元组里第五个元素num_completion_tokens装的是[列表] (完整的 token_ids)。
            # 如果是 Decode 阶段：元组里第五个元素里装的是 整数 (last_token)

    def __setstate__(self, state):
        # 触发时机：当使用 pickle.load() 反序列化一个对象时。 默认行为：如果没有定义这个函数，Python 会直接把解包出来的字典更新到 self.__dict__ 里。
        # 自定义作用：“我要自己决定怎么还原。”你可以根据解包出来的数据，重新计算一些属性。你可以处理版本兼容性问题。
        # 参数 state：这就是刚才 __getstate__ 返回的那个东西
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1] #取出元组里除了最后一个元素之外的所有东西（即前4个）
        if self.num_completion_tokens == 0: #没有新生成的token，即现在是 Prefill 阶段 (刚开始)，而不是 Decode 阶段
            self.token_ids = state[-1] #取出元组的最后一个元素，因为是prefill阶段所以 state[-1] 里装的是 完整的列表
        else:  #否则是decode阶段
            self.last_token = state[-1] #因为是 Decode，所以 state[-1] 里装的是单个整数，这行代码是在更新最新的那个字
            
"""
在 nano-vllm 中，这两个函数被用来做极致的性能优化。

背景：主进程（Engine）需要把 Sequence 对象传给子进程（ModelRunner）。
痛点：Sequence 里有一个 token_ids 列表。在长文本生成时，这个列表可能包含几千个整数。如果每生成一个字，主进程都要把这几千个整数打包发给子进程，通信带宽会被撑爆，速度会变慢。

作者利用这两个钩子实现了“增量传输”：
代码段 1：打包 (__getstate__)
作用：根据当前是“刚开始”还是“生成中”，动态决定是寄送“整个集装箱”还是只寄送“一个零件”。

代码段 2：拆包 (__setstate__)
作用：子进程收到包裹后，看一眼数据。如果是完整的列表，就存好；如果是单个整数，就只更新 last_token。

总结
钩子函数：允许你“插手”系统默认流程的函数。
__getstate__：“打包清单”。告诉 pickle 哪些数据需要保存/发送。
__setstate__：“还原指南”。告诉 pickle 如何利用收到的数据重建对象。
Nano-vLLM 的用法：利用这两个钩子实现了按需序列化，极大地减少了多进程通信的数据量，这是分布式系统中非常高级且实用的技巧
"""