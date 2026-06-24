# The Annotated Transformer 中文源码精读：从图解论文到可运行代码

这份笔记重写自 Harvard NLP 的 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)，并使用官方 GitHub 中的 `the_annotated_transformer.py` 源码作为讲解对象。图片已保存到 `images/annotated-transformer/`，源码块按原文代码顺序保留，讲解部分用中文重新组织，不做英文逐字翻译。

读这篇时建议同时打开 [`illustrated-transformer-cn.md`](illustrated-transformer-cn.md)。前一篇偏“看图理解 Transformer 是什么”，这一篇偏“看代码理解 Transformer 怎么跑起来”。最重要的学习路线是：

1. 先知道输入文本会变成 token id，再变成向量。
2. 再看 encoder 如何用 self-attention 理解源句子。
3. 再看 decoder 如何用 mask 和 source attention 生成目标句子。
4. 最后看 loss、梯度下降、学习率、数据加载和可视化如何把论文模型训练成可用系统。

![Attention is All You Need](https://cdn.jsdelivr.net/gh/13212133871/LLMLeran@main/images/annotated-transformer/01-attention-is-all-you-need.png)

## 0. 先把两篇文章对齐

| Illustrated Transformer 里的图解概念 | Harvard 源码里的对应位置 | 一句话理解 |
| --- | --- | --- |
| Transformer 总览 | `EncoderDecoder` | 把编码器、解码器、嵌入层、输出层装成一台机器 |
| 编码器堆叠 | `Encoder`、`EncoderLayer` | 多层反复理解源句子 |
| 解码器堆叠 | `Decoder`、`DecoderLayer` | 一边看已生成内容，一边看源句子 |
| Self-Attention | `attention(query, key, value)` | 每个词按权重收集其他词的信息 |
| Q/K/V | `linear(x).view(...).transpose(...)` | 同一个词向量投影成“查询、索引、内容”三种角色 |
| 多头注意力 | `MultiHeadedAttention` | 多个视角同时看句子关系 |
| 残差连接 | `x + sublayer(...)` | 保留原信息，再叠加新加工结果 |
| LayerNorm | `LayerNorm` | 稳定每层数字范围，让训练更顺 |
| 前馈网络 | `PositionwiseFeedForward` | 每个位置独立做非线性加工 |
| 词嵌入 | `Embeddings` | token 编号查表变向量 |
| 位置编码 | `PositionalEncoding` | 给没有循环结构的 Transformer 加顺序感 |
| 训练 | `run_epoch`、`SimpleLossCompute` | 预测、算错、反向传播、更新参数 |
| 解码 | `greedy_decode` | 生成时一步一步选下一个 token |

下面这张结构图是原文核心。先不要急着记每个箭头，只要抓住：左边 encoder 读输入，右边 decoder 写输出，中间注意力负责信息流动。

![Transformer 架构](https://cdn.jsdelivr.net/gh/13212133871/LLMLeran@main/images/annotated-transformer/02-transformer-architecture.png)


## 一、准备运行环境：先把工具箱摆好

这一部分不是 Transformer 的数学核心，但它决定代码能不能跑起来。可以把它想象成做实验之前先准备纸、笔、计算器和材料。没有这些工具，后面的模型再漂亮也只是图纸。

### 1. 安装依赖：代码运行前的工具箱

原文位置：`Prelims`；这段主要包含 `# # !pip install -r requirements.txt`。

```python

# # !pip install -r requirements.txt

```

这段安装命令告诉读者：原文不是一篇只讲概念的博客，它本身是一个可以运行的 notebook。`requirements.txt` 里会列出 PyTorch、torchtext、spacy、altair 等依赖。

用生活类比说，Transformer 是一台机器，代码是说明书，但运行它还要有螺丝刀和零件。安装依赖就是先把工具箱装齐。你不需要先懂每个工具的内部原理，只要知道 PyTorch 负责张量和自动求导，spacy 负责分词，altair 负责画图，torchtext 负责把文本数据整理成模型能吃的批次。

### 2. Colab 环境：在线 notebook 怎么准备

原文位置：`Prelims`；这段主要包含 `# # Uncomment for colab`。

```python

# # Uncomment for colab
# #
# # !pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
# # !python -m spacy download de_core_news_sm
# # !python -m spacy download en_core_web_sm

```

这一块是给 Colab 环境准备的安装命令。Colab 是 Google 提供的在线 Python 笔记本，很多读者不想在自己电脑上折腾 GPU、CUDA、环境变量，就可以直接在网页里跑。

这里列出的 `de_core_news_sm` 和 `en_core_web_sm` 是德语、英语的 spacy 分词模型。翻译任务第一步不是直接把整句话塞进神经网络，而是先把句子切成 token。比如 `I love apples` 会先切成 `I`、`love`、`apples`，德语句子也一样。模型真正看到的是 token 的编号，而不是人眼看到的文字。

### 3. 导入库：PyTorch、数据处理和画图工具

原文位置：`Prelims`；这段主要包含 `import os`。

```python

import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

```

这一块把后面要用的库都导入进来。`torch` 是 PyTorch 主库，`nn` 提供神经网络层，`DataLoader` 负责按批次喂数据，`math` 负责公式计算，`altair` 和 `pandas` 负责画图和整理表格。

对零基础读者来说，`import` 可以理解成“把别人已经写好的工具拿到桌面上”。比如你要算平方根，不必自己造一个平方根算法，可以直接调用数学库；你要做线性层、Dropout、LayerNorm，也不必从最底层矩阵乘法开始写，PyTorch 已经封装好了。

### 4. 辅助函数：让 notebook 演示更顺滑

原文位置：`Prelims`；这段主要包含 `def is_interactive_notebook`、`def show_example`、`def execute_example`、`class DummyOptimizer`。

```python

# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

```

这段写了一些辅助函数和两个“假优化器”。它们的作用不是定义 Transformer，而是让 notebook 在不同环境下更顺滑地演示。

`show_example` 和 `execute_example` 的思路很朴素：有些函数是为了展示结果，有些环境是交互式 notebook，有些环境是脚本执行，所以原文用这些小函数统一处理。`DummyOptimizer` 和 `DummyScheduler` 是占位工具，在只想演示流程、不想真的更新参数时使用。

从学习角度看，你可以先把它们放在“杂务代码”这个抽屉里。真正的 Transformer 原理从 `EncoderDecoder` 开始。

## 二、模型骨架：从整台机器看到编码器和解码器

这一部分对应 `illustrated-transformer-cn.md` 里“Encoder-Decoder 总览”“残差连接”“LayerNorm”“mask”等章节。Jay Alammar 的文章用图解释组件长什么样，Harvard 这篇把组件写成 PyTorch 类。读源码时先记住一个大原则：每个类都像工厂里的一个小车间，输入一批数字，输出另一批更有信息的数字。

### 5. EncoderDecoder：Transformer 的总入口

原文位置：`Model Architecture`；这段主要包含 `class EncoderDecoder`。

```python

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

```

`EncoderDecoder` 是整台 Transformer 的外壳。它把五个大部件装在一起：编码器 `encoder`、解码器 `decoder`、源语言嵌入 `src_embed`、目标语言嵌入 `tgt_embed`，以及最终把向量变成词概率的 `generator`。

先把这类 PyTorch 代码的固定写法看懂，后面几乎所有类都会重复这个模式：

```python
class 某个网络层(nn.Module):
    def __init__(self, ...):
        super(...).__init__()
        self.零件1 = ...
        self.零件2 = ...

    def forward(self, 输入):
        输出 = self.零件2(self.零件1(输入))
        return 输出
```

`nn.Module` 是 PyTorch 里所有神经网络模块的基类。你可以把它理解成“神经网络零件的标准外壳”。只要一个类继承了 `nn.Module`，PyTorch 就知道它可能包含参数、子模块、训练/推理状态，也知道怎么把它放到 GPU、怎么保存权重、怎么递归找到里面所有可学习参数。

`__init__` 是“装配阶段”，通常只在创建对象时运行一次。比如 `make_model()` 里创建 `EncoderDecoder(...)` 时，`__init__` 会把 encoder、decoder、embedding、generator 都挂到 `self` 上。重点是：当你写下 `self.encoder = encoder`、`self.proj = nn.Linear(...)` 这种赋值时，PyTorch 不只是普通保存变量，还会把这些子模块登记到模型内部。以后调用 `model.parameters()` 时，PyTorch 就能沿着这些登记信息找到所有参数。

`super(EncoderDecoder, self).__init__()` 的作用，是先把 `nn.Module` 自己内部的登记表初始化好。你可以想象 PyTorch 先准备一本空账本，后面你把 `self.encoder`、`self.decoder`、`self.generator` 挂上去，它才能记账。如果漏掉 `super().__init__()`，很多模块和参数就无法被 PyTorch 正确管理。

`forward()` 是“计算阶段”，每来一批数据就会运行一次。它不负责创建零件，而是规定数据怎么从一个零件流到下一个零件。比如这里：

```python
return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
```

数据流是：

$$
src\rightarrow src\_embed\rightarrow encoder\rightarrow memory
$$

$$
tgt\rightarrow tgt\_embed\rightarrow decoder(memory)\rightarrow decoder隐藏向量
$$

平时我们通常写：

```python
out = model(src, tgt, src_mask, tgt_mask)
```

而不是直接写：

```python
out = model.forward(src, tgt, src_mask, tgt_mask)
```

原因是 `model(...)` 会先进入 `nn.Module` 已经写好的 `__call__` 逻辑，再由它调用你定义的 `forward()`。这个外层逻辑会处理 hook、训练/推理状态等 PyTorch 机制。初学时可以先简单记住：你写 `forward()`，但调用模型时写 `model(...)`。

反向传播也和 `forward()` 密切相关，但不是说你要手写一个 `backward()`。PyTorch 的自动求导机制会在 `forward()` 运行时记录张量经历了哪些运算。只要这些张量和模型参数有关，并且参数默认 `requires_grad=True`，PyTorch 就会搭出一张“计算图”。

训练时大概是这样：

```python
out = model(src, tgt, src_mask, tgt_mask)
log_probs = model.generator(out)
loss = criterion(log_probs, tgt_y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

其中 `loss.backward()` 会沿着刚才 `forward()` 产生的计算图，从 loss 倒着往前算每个参数应该怎么改。比如 loss 发现 `student` 的概率给低了，它会一路把责任传回 `Generator`、decoder、attention、embedding 等相关参数，算出每个参数的梯度。`optimizer.step()` 再按照这些梯度真正修改参数。

可以把它类比成一条流水线：`forward()` 是产品从原料到成品的路线；`loss` 是质检发现成品哪里不好；`backward()` 是沿流水线倒查每个机器该调多少；`optimizer.step()` 是工人真的去拧螺丝。你不用手写每颗螺丝怎么拧，PyTorch 根据计算图自动算。

再换成一个完整的“翻译工厂”类比，会更直观。

`class EncoderDecoder(nn.Module)` 就像在定义一种工厂类型：这不是某一家具体工厂，而是一张“Transformer 翻译工厂”的建造图纸。只要按这张图纸建出来的工厂，都有编码车间、解码车间、词向量入口、最终出词口。

`__init__` 是建厂和安装设备。你把 `encoder`、`decoder`、`src_embed`、`tgt_embed`、`generator` 交给它，它就把这些设备固定到这家工厂里：

```python
self.encoder = encoder
self.decoder = decoder
self.src_embed = src_embed
self.tgt_embed = tgt_embed
self.generator = generator
```

这里的 `self` 可以理解成“这家工厂自己”。`self.encoder = encoder` 的意思是：把编码器这台机器装到这家工厂身上，以后这家工厂运行时就能说“使用我自己的编码器”。这比普通变量重要，因为 PyTorch 会登记这些设备里面的参数。登记以后，训练时才能统一找到它们、保存它们、更新它们。

`forward` 是工厂的生产流程，不是建厂流程。建厂只做一次，但生产流程每来一批句子就跑一次。比如来了源句子 `je suis étudiant`，工厂不是重新安装 encoder，而是沿着已经装好的机器走一遍：

1. `src_embed(src)`：把源语言 token 编号换成向量，好比把原料贴上机器能识别的数字标签。
2. `encoder(...)`：编码车间读完整句源语言，做成一份“源句子理解报告”，代码里叫 `memory`。
3. `tgt_embed(tgt)`：把已经生成的目标语言 token 也换成向量，比如已经有 `<s> I am a`。
4. `decoder(...)`：解码车间一边看目标语言草稿，一边看源句子理解报告，输出当前位置的隐藏向量。
5. `generator(...)`：最后出词口把隐藏向量变成词表概率，比如 `student` 概率最高。

第 5 步没有放在 `EncoderDecoder.forward()` 里，是因为这家工厂有两种使用方式。训练时，可能要把所有位置的隐藏向量一次性送去 `generator` 算 loss；生成时，只需要把最后一个位置的隐藏向量送去 `generator` 预测下一个词。所以原文把 `generator` 装在工厂里，但不强行规定每次 `forward()` 都立刻出词。

再看自动反向传播。假设工厂输出错了，把 `student` 的概率给低了，`loss` 就像质检报告：这次产品不合格，差距是多少。`loss.backward()` 不是重新跑一遍翻译，而是沿着刚才生产时留下的记录倒查：最终概率来自 `generator`，`generator` 的输入来自 decoder，decoder 的信息来自 attention、feed-forward、embedding、encoder。PyTorch 会一站一站倒着算：每台机器的每个旋钮，对最终错误贡献了多少。

这个“每个旋钮该往哪个方向调、调多少”的数，就叫梯度。梯度不是参数本身，而是修改建议。`optimizer.step()` 才是真正按照建议调整参数：

$$
\theta_{\text{new}}=\theta_{\text{old}}-\eta\cdot\nabla_{\theta}\text{loss}
$$

这里的 $$\theta$$ 可以代表任何参数，比如 `Generator` 里的线性层权重、Q/K/V 矩阵、embedding 表。$$\eta$$ 是学习率，表示每次听取建议时迈多大一步。

用更小的例子看，假设只有一个简单模型：

```python
y = ax + b
```

`__init__` 就是准备参数 `a` 和 `b`；`forward(x)` 就是计算 `ax+b`；`loss` 比较预测值和真实值差多少；`backward()` 算出如果想让 loss 变小，`a` 和 `b` 分别该增大还是减小；`optimizer.step()` 真的更新 `a` 和 `b`。Transformer 只是把 `a`、`b` 换成了大量矩阵，但学习逻辑没有变。

再往底层看，PyTorch 自动反向传播靠的是 `autograd`，可以理解成“边计算边记账”的系统。它不是提前把整张网络图写死，而是在 `forward()` 真正运行时，动态记录这一次计算经过了哪些操作。这种方式通常叫动态图：你这次代码走了哪些分支，PyTorch 就记录哪些分支。

第一步，参数张量会带着 `requires_grad=True`。比如 `nn.Linear` 里的权重 `weight` 和偏置 `bias`，默认就是需要梯度的。它们是要被训练改变的数字，所以 PyTorch 会盯着它们参与过哪些计算。

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
y = a * x + b
loss = (y - 10) ** 2
loss.backward()
```

这段里，`a` 和 `b` 是可学习参数，`x` 是普通输入，`y` 是预测，`loss` 是错误大小。运行 `y = a * x + b` 时，PyTorch 会发现：`y` 不是凭空来的，它由乘法和加法得到，并且依赖 `a`、`b`。于是 `y` 背后会挂着一个 `grad_fn`，意思是“如果以后有人问我怎么反传，我知道自己是怎么被算出来的”。

可以把这次计算画成一张小图：

$$
a,x \rightarrow a\cdot x
$$

$$
a\cdot x,b \rightarrow y=a\cdot x+b
$$

$$
y,10 \rightarrow loss=(y-10)^2
$$

这张图从输入到 loss 是正向计算图。`backward()` 做的事情，就是从最后的 `loss` 出发，沿着图反方向走回去，把导数一路传给每个参数。

为什么能反方向传？核心是链式法则。对于上面这个简单模型：

$$
y=ax+b
$$

$$
loss=(y-t)^2
$$

其中 $$t$$ 是真实答案。我们想知道 `a` 和 `b` 怎么影响 loss，就要求：

$$
\frac{\partial loss}{\partial a}
=
\frac{\partial loss}{\partial y}
\cdot
\frac{\partial y}{\partial a}
$$

$$
\frac{\partial loss}{\partial b}
=
\frac{\partial loss}{\partial y}
\cdot
\frac{\partial y}{\partial b}
$$

如果 $$a=2$$、$$b=1$$、$$x=3$$、$$t=10$$，那么：

$$
y=2\cdot3+1=7
$$

$$
loss=(7-10)^2=9
$$

先算 loss 对 y 的导数：

$$
\frac{\partial loss}{\partial y}=2(y-t)=2(7-10)=-6
$$

再算 y 对参数的导数：

$$
\frac{\partial y}{\partial a}=x=3
$$

$$
\frac{\partial y}{\partial b}=1
$$

所以：

$$
\frac{\partial loss}{\partial a}=-6\cdot3=-18
$$

$$
\frac{\partial loss}{\partial b}=-6\cdot1=-6
$$

这两个数就是梯度。它们会被 PyTorch 存到：

```python
a.grad
b.grad
```

如果学习率 $$\eta=0.01$$，优化器更新时大概就是：

$$
a_{\text{new}}=a_{\text{old}}-0.01\cdot(-18)=2.18
$$

$$
b_{\text{new}}=b_{\text{old}}-0.01\cdot(-6)=1.06
$$

为什么是减去梯度？因为梯度指向 loss 增长最快的方向，想让 loss 下降，就往反方向走。这里梯度是负数，所以减去负数等于增大参数。增大 `a` 和 `b` 后，预测值 `y` 会从 7 往 10 靠近。

PyTorch 不是为每个大模型手写这些导数，而是每个基础操作都知道自己的局部反向公式。乘法知道怎么反传，加法知道怎么反传，矩阵乘法知道怎么反传，softmax、LayerNorm、ReLU 也都有对应的反向规则。`forward()` 时这些操作像一串小节点被连起来；`backward()` 时 PyTorch 从 loss 节点开始，按相反顺序调用每个节点的反向规则。

比如 ReLU 的正向是：

$$
\text{ReLU}(x)=\max(0,x)
$$

它的反向规则很简单：如果正向时 $$x>0$$，梯度原样传过去；如果 $$x<0$$，梯度变成 0。因为负数区域被 ReLU 截断了，那里输出不随输入变化。

矩阵乘法也是一样。假设：

$$
Y=XW
$$

反向传播时，如果已经知道 $$\frac{\partial loss}{\partial Y}$$，就可以继续算出：

$$
\frac{\partial loss}{\partial X}
=
\frac{\partial loss}{\partial Y}W^T
$$

$$
\frac{\partial loss}{\partial W}
=
X^T\frac{\partial loss}{\partial Y}
$$

Transformer 里的 Q/K/V 线性层、前馈网络、`Generator` 输出层，本质上都有大量矩阵乘法。PyTorch 自动求导就是把这些矩阵乘法的反向公式一层层串起来。

还要注意一个细节：梯度会累积。也就是说，如果你连续调用两次 `loss.backward()` 而不清空，`a.grad` 会把两次梯度加在一起。所以训练循环里常见：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`zero_grad()` 是先把上一次的梯度清掉；`backward()` 是根据这一次 batch 重新计算梯度；`step()` 是用这一次梯度更新参数。原文有些地方把 `zero_grad()` 放在 `step()` 后面，本质也是为了保证下一轮开始前梯度是干净的。

把这套机制放回 Transformer，就是：

1. `forward()` 从 token id 一路算到 decoder 隐藏向量。
2. `generator` 把隐藏向量算成词表概率。
3. `loss` 比较词表概率和正确 token。
4. `loss.backward()` 沿着计算图反向经过 `Generator`、decoder、attention、encoder、embedding。
5. 每个参数的 `.grad` 里得到“这次该怎么改”的建议。
6. `optimizer.step()` 更新这些参数。

所以 PyTorch 自动化的关键不是“它懂 Transformer”，而是它懂每个基础数学操作的导数，并且能在 `forward()` 运行时把这些操作连成计算图。Transformer 再复杂，也只是很多基础操作搭出来的巨大计算图。

后文每个继承 `nn.Module` 的类，都可以用“`__init__` 装零件，`forward` 跑数据”这个框架来读：

| 类 | `__init__` 主要装什么 | `forward()` 主要做什么 |
| --- | --- | --- |
| `EncoderDecoder` | encoder、decoder、源/目标 embedding、generator | 先 encode 源句子，再 decode 目标句子，返回 decoder 隐藏向量 |
| `Generator` | 一个 `Linear(d_model, vocab)` | 把隐藏向量变成词表上的 log 概率 |
| `Encoder` | N 个 `EncoderLayer` 和最终 LayerNorm | 让输入依次通过 N 层 encoder |
| `LayerNorm` | 可学习的缩放参数和偏移参数 | 对每个 token 向量做归一化 |
| `SublayerConnection` | LayerNorm 和 Dropout | 做残差连接：原输入加上子层输出 |
| `EncoderLayer` | self-attention、feed-forward | 先让词互相看，再逐位置加工 |
| `Decoder` | N 个 `DecoderLayer` 和最终 LayerNorm | 让目标序列依次通过 N 层 decoder |
| `DecoderLayer` | masked self-attention、source-attention、feed-forward | 先看已生成目标词，再看源句子，再加工 |
| `MultiHeadedAttention` | 4 个线性层和 dropout | 生成 Q/K/V，多头注意力，拼接后再线性输出 |
| `PositionwiseFeedForward` | 两个线性层 | 对每个位置做 `Linear -> ReLU -> Linear` |
| `Embeddings` | embedding 查表矩阵 | 把 token id 查成向量 |
| `PositionalEncoding` | 预先算好的位置编码表 | 把位置编码加到词向量上 |
| `LabelSmoothing` | KLDivLoss 和平滑参数 | 把硬标签变柔和，再算损失 |

所以后面看到任何 `forward()`，都先问三个问题：输入是什么形状？中间经过哪些子模块？输出交给谁？只要这三件事清楚，源码主线就不会乱。

这里有一个很容易困惑的地方：`__init__` 里确实保存了 `self.generator = generator`，但 `forward()` 里没有调用它。也就是说，这个类的 `forward()` 默认只跑到 decoder 的输出为止，返回的是“每个目标位置的隐藏向量”，还没有变成“词表概率”。

为什么这样设计？因为训练和推理阶段使用 `generator` 的方式不完全一样。训练时通常会一次性把所有位置的 decoder 输出送进 `generator`，再计算 loss；推理时常常只拿最后一个位置的输出送进 `generator`，因为我们只需要预测“下一个词”。所以原文把 `generator` 挂在模型身上，但让训练循环或解码函数在合适的位置调用它。

你可以在后文看到三处真正调用：

```python
prob = test_model.generator(out[:, -1])
```

第 23 节的 `inference_test()` 里，模型每生成一步，只拿最后一个位置 `out[:, -1]` 去预测下一个 token。

```python
x = self.generator(x)
```

第 33 节的 `SimpleLossCompute` 里，训练时把 decoder 输出交给 `generator`，得到词概率后再算 loss。

```python
prob = model.generator(out[:, -1])
```

第 34 节的 `greedy_decode()` 里，贪心解码也是每一步只看最后位置，然后选概率最大的下一个 token。

所以第 5 节不是 `generator` 没用，而是这里先把它注册为模型的一部分。真正“打开最后输出口”的动作，放到了训练和解码流程里。

一次翻译可以理解成两步。第一步，编码器读完整个源句子，比如德语句子，把每个词加工成带上下文的信息；第二步，解码器一边看编码器结果，一边看自己已经生成的英文词，继续预测下一个英文词。

在代码里，`forward` 的公式路线更准确地说是：

$$
\text{decoder隐藏向量}=\text{decode}(\text{encode}(\text{src}),\text{tgt})
$$

如果还要得到词概率，需要再接一步：

$$
\text{词表概率}=\text{generator}(\text{decoder隐藏向量})
$$

这和 Illustrated Transformer 里的大图完全对应：左边一叠 encoder 先理解输入，右边一叠 decoder 先产出隐藏向量，最后的 Linear + Softmax 再把隐藏向量变成具体 token 的概率。

### 6. Generator：把隐藏向量变成词概率

原文位置：`Model Architecture`；这段主要包含 `class Generator`。

```python

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

```

`Generator` 是最后一步：把 decoder 产出的隐藏向量变成“下一个 token 可能是谁”的概率。要理解这段代码，先把几个词拆开。

`token` 是模型处理文本的基本单位。它可以是一个完整单词，也可以是一个子词、标点或特殊符号。例如一个极简目标语言词表可能是：

| token | 编号 |
| --- | --- |
| `<s>` | 0 |
| `</s>` | 1 |
| `<blank>` | 2 |
| `<unk>` | 3 |
| `I` | 4 |
| `am` | 5 |
| `a` | 6 |
| `student` | 7 |

这张“token 到编号”的字典就叫词表。`vocab` 不是某个单词，而是词表大小，也就是候选答案一共有多少种。上面这个玩具词表里有 8 个 token，所以 `vocab=8`。真实翻译模型里，目标语言词表可能有几万项，比如 `vocab=30000`。

`d_model` 是 Transformer 内部每个 token 向量的长度。论文 base 模型常用 `d_model=512`，意思是：每个位置不是用一个数字表示，而是用 512 个数字共同描述。你可以把它想成一张很长的“特征表”：某些维度可能逐渐学到语法信息，某些维度可能学到语义信息，某些维度可能和位置、搭配、时态有关。模型不会把这些维度命名成人类可读的标签，但训练会让这些数字越来越有用。

为了看清楚，先用 `d_model=4` 的小例子。假设 decoder 在某个位置输出的隐藏向量是：

$$
x=[0.2,-0.5,1.1,0.7]
$$

这 4 个数字不是概率，也不是 token 编号，而是模型对当前位置的内部总结。它可能总结了“前面已经生成了 I am a”“源句子里有 étudiant”“现在大概率该生成 student”这些信息。

`nn.Linear(d_model, vocab)` 的意思是：把长度为 `d_model` 的向量，变成长度为 `vocab` 的打分表。如果 `d_model=4`、`vocab=8`，它就把 4 个数字变成 8 个数字：

$$
[0.2,-0.5,1.1,0.7]\rightarrow[分给<s>,分给</s>,分给<blank>,分给<unk>,分给I,分给am,分给a,分给student]
$$

真实模型里如果 `d_model=512`、`vocab=30000`，就是把 512 个内部特征变成 30000 个候选 token 的分数。每个候选 token 都会得到一个分数，这些分数还不是概率，通常叫 `logits`。可以粗略理解成“还没归一化的原始打分”。

`self.proj = nn.Linear(d_model, vocab)` 里面有一组要学习的权重和偏置。它本质上是在给每个候选 token 准备一套打分规则。比如 `student` 这个 token 会有自己的一行权重，拿这行权重和当前隐藏向量做匹配；匹配越强，`student` 的分数越高。

如果词表里有 30000 个 token，那么每个位置都会得到 30000 个分数。分数越高，代表模型越倾向于选那个 token。代码里的 `log_softmax` 会把这些原始分数变成对数概率，训练时更稳定，也更方便和 loss 函数配合。

数学上可以写成：

$$
z=xW+b
$$

$$
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

这里的 `x` 是 decoder 给出的向量，`W` 和 `b` 是要学习的参数，`P_i` 是第 `i` 个词的概率。你可以把它想成最后的选择器：前面所有层都在整理信息，最后这一层把信息翻译成词表上的概率。

举一个完整的小流程。假设目标词表只有 8 个 token，模型正在翻译 `je suis étudiant`，前面已经生成了 `I am a`，现在 decoder 输出一个隐藏向量 `x`。`Generator` 会给 8 个 token 都打分，再经过 `log_softmax` 变成概率。最后可能得到类似：

| token | 概率 |
| --- | --- |
| `<s>` | 0.01 |
| `</s>` | 0.02 |
| `<blank>` | 0.00 |
| `<unk>` | 0.01 |
| `I` | 0.03 |
| `am` | 0.04 |
| `a` | 0.09 |
| `student` | 0.80 |

这时贪心解码就会选 `student`。训练时如果正确答案确实是 `student`，loss 会变小；如果模型给 `student` 的概率很低，loss 会变大，梯度下降就会调整 `Generator`、decoder、attention、embedding 等相关参数，让下次更容易给正确 token 高分。

### 7. clones：复制多层但让每层参数独立

原文位置：`Encoder`；这段主要包含 `def clones`。

```python

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

```

`clones(module, N)` 会复制出 `N` 个结构相同但参数独立的层。Transformer 论文里 encoder 有 6 层，decoder 也有 6 层；这段代码就是为了方便生成“一模一样的层结构”。

注意“结构相同”和“参数相同”不是一回事。就像同一所学校有 6 个年级，每个年级都有语文课、数学课、英语课，但每个年级老师讲的深度不一样。这里用 `copy.deepcopy`，意味着每一层有自己的权重，训练后会学到不同功能。

### 8. Encoder：把多个编码层串起来

原文位置：`Encoder`；这段主要包含 `class Encoder`。

```python

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

```

`Encoder` 是多个 `EncoderLayer` 串起来。输入先过第 1 层，再过第 2 层，一直过到第 N 层，最后再做一次 LayerNorm。

如果把输入记作 $$X^{(0)}$$，第 $$l$$ 层可以粗略写成：

$$
X^{(l+1)}=\text{EncoderLayer}^{(l)}(X^{(l)})
$$

每多过一层，词向量就多吸收一轮上下文。第一层可能主要学附近词的关系，后面层可能学更抽象的语义关系。比如 `bank` 在句子里是“银行”还是“河岸”，不是单靠它自己决定，而是不断结合周围词后逐层变清楚。

### 9. LayerNorm：让每层数字保持稳定

原文位置：`Encoder`；这段主要包含 `class LayerNorm`。

```python

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

```

`LayerNorm` 是归一化层。它会对每个 token 的向量内部做标准化，让这一组数字的均值接近 0、方差接近 1，然后再乘上可学习的缩放参数 `a_2`，加上可学习的平移参数 `b_2`。

公式是：

$$
\text{LayerNorm}(x)=a\cdot\frac{x-\mu}{\sigma+\epsilon}+b
$$

为什么要这么做？神经网络训练时，数字如果一会儿特别大、一会儿特别小，后面的层会很难适应。归一化就像把不同音量的录音先调到差不多大小，再交给后面的设备处理。`a` 和 `b` 又允许模型在需要时把尺度调回来，所以不是死板地强行压平。

### 10. SublayerConnection：残差连接、归一化和 Dropout

原文位置：`Encoder`；这段主要包含 `class SublayerConnection`。

```python

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

```

`SublayerConnection` 可以理解成“给每个重要子层套上的保护壳”。Transformer 里的 self-attention、source-attention、feed-forward 都不是直接裸跑，而是经常被这个结构包起来：

```python
return x + self.dropout(sublayer(self.norm(x)))
```

这行代码从里到外看，顺序是：

1. 先对输入 `x` 做 LayerNorm。
2. 把归一化后的结果送进某个子层 `sublayer`。
3. 对子层输出做 Dropout。
4. 最后把结果加回原始输入 `x`，形成残差连接。

写成公式就是：

$$
y=x+\text{Dropout}(\text{Sublayer}(\text{LayerNorm}(x)))
$$

这里的 `sublayer` 不是某一个固定层，而是一个“传进来的函数”。在第 11 章里，它可以是 self-attention：

```python
lambda x: self.self_attn(x, x, x, mask)
```

也可以是 feed-forward：

```python
self.feed_forward
```

所以 `SublayerConnection` 的设计很聪明：它不关心里面包的是注意力还是前馈网络，只负责给这个子层统一加上归一化、Dropout 和残差连接。

先讲残差连接。`x + ...` 这部分就是残差连接。它的意思是：子层可以往原信息上“添加新理解”，但不能轻易把原信息完全覆盖掉。

假设某个 token 的输入向量 `x` 可以想象成一张原始笔记：

```text
原始笔记 x：这个词是 it，可能是代词，位置在句子后半段
```

self-attention 子层会补充一些新信息：

```text
子层新增信息：it 可能指 animal，而不是 street
```

残差连接做的是把两者加起来：

```text
输出：保留“it 自己是谁”的信息，同时加入“它可能指 animal”的信息
```

如果没有残差连接，每一层都可能把前一层的信息全部重写。层数一多，早期信息容易丢，梯度也更难一路传回前面的层。有了残差连接，至少有一条很直接的通路：

$$
y=x+\text{新增信息}
$$

这条通路让信息和梯度都更容易穿过很多层。直观说，模型不是每层都从零开始重写句子理解，而是在原来的理解上逐层做批注。

再讲这里的“前置归一化”，也就是英文里常说的 pre-norm。中文可以理解成“先归一化，再进子层”。这份代码的顺序是：

$$
y=x+\text{Dropout}(\text{Sublayer}(\text{LayerNorm}(x)))
$$

这叫前置归一化。另一种写法叫后置归一化，大概是：

$$
y=\text{LayerNorm}(x+\text{Dropout}(\text{Sublayer}(x)))
$$

后置归一化更接近原论文图里的 “Add & Norm” 画法；这份 Harvard 代码采用的是前置归一化，也就是先把 `x` 调整到比较稳定的数值范围，再交给 self-attention 或 feed-forward。

为什么要先归一化？可以用“送材料进机器”来理解。假设一个机器平时适合处理大小比较稳定的材料，结果输入有时特别大、有时特别小，它就很难学出稳定规则。LayerNorm 先把每个 token 向量内部的数字拉到相对稳定的范围，相当于先把材料规格统一，再送进机器加工。

前置归一化还有一个很重要的好处：残差路径更顺。注意公式里最外面保留了一个干净的 `x + ...`：

$$
y=x+\text{某个加工结果}
$$

这意味着即使子层很深，梯度也可以沿着这个加法里的 `x` 比较直接地往回走。对于堆很多层的 Transformer，这通常会让训练更稳定。你可以把它想象成楼梯旁边有一条直达扶梯，信息和梯度不必每次都绕过复杂机器才能往前、往后传。

不过要注意：前置归一化不是说“后置归一化一定错”。它们是两种结构选择。原论文图更像后置归一化，很多现代 Transformer 实现更偏向前置归一化，因为深层模型训练时往往更稳。

最后讲 Dropout。`self.dropout = nn.Dropout(dropout)` 里的 `dropout` 通常是一个概率，比如第 22 章默认 `dropout=0.1`。意思是训练时随机把一部分输出置为 0。注意是训练时随机，不是永久删除；推理时 `model.eval()` 后 Dropout 会关闭。

举一个小数字例子。假设某个子层输出是：

$$
[2,\ 1,\ -3,\ 4]
$$

如果 Dropout 概率设成 $$0.5$$，训练时可能随机生成一个遮罩：

$$
[1,\ 0,\ 1,\ 0]
$$

那么第 2、4 个位置会被临时丢掉。PyTorch 为了保持整体期望大小，训练时还会把保留下来的值按比例放大。所以上面可能变成：

$$
[4,\ 0,\ -6,\ 0]
$$

如果 Dropout 概率是 $$0.1$$，就表示平均随机丢掉 10% 的输出。每次训练 batch 的随机位置都可能不一样。

Dropout 为什么能减少死记硬背？假设模型在训练集中发现一个偷懒规律：只要某个维度特别大，就总是预测某个词。没有 Dropout 时，它可能过度依赖这个单一线索；有 Dropout 后，这个线索有时会被临时遮掉，模型就被迫同时学习其他线索。就像考试复习时，有些提示卡片会随机被拿走，你不能只背一张卡片，必须真正理解多个证据之间的关系。

放到 Transformer 里，Dropout 可能作用在注意力输出、前馈网络输出、位置编码后等地方。比如 self-attention 刚刚给 `it` 混入了来自 `animal` 的信息，Dropout 可能随机遮掉其中一部分维度。这样模型不能只依赖某一两个维度表达“it 指 animal”，而要把这种关系分散、稳健地学在多个维度和多个头里。

所以这一行代码的整体含义可以这样记：

```text
先把输入数字调稳
        ↓
交给注意力或前馈网络加工
        ↓
训练时随机遮掉一部分加工结果，防止依赖单一线索
        ↓
把加工结果加回原始输入，保留原信息
```

`SublayerConnection` 的名字里有 `Connection`，重点就在这个“连接”：它把原始信息、子层新信息、训练稳定性、防过拟合这几件事接在一起。

### 11. EncoderLayer：自注意力加前馈网络

原文位置：`Encoder`；这段主要包含 `class EncoderLayer`。

```python

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

```

`EncoderLayer` 是编码器里反复堆叠的一个基本单元。Transformer 的 encoder 不是只做一次理解，而是把同样结构的层堆很多次。每一层都做两件事：先让句子里的词互相交流，再让每个词把交流后的信息整理一遍。

先用一个日常类比。假设一句话里每个词都是会议室里的一个人：

```text
The animal didn't cross the street because it was too tired
```

一开始，每个人只拿着自己的身份牌。`animal` 知道自己大概是“动物”，`street` 知道自己是“街道”，`it` 只知道自己是一个代词，但还不知道自己指谁。EncoderLayer 的第一步 self-attention 就像开会：每个人可以去听其他人说话，并决定自己应该重点听谁。`it` 可能会重点听 `animal`，因为句子里“太累”的更可能是动物，不太可能是街道。

开完会以后，每个人手里的笔记就不再只是自己的信息，而是混入了别人的信息。`it` 的新笔记里会有更多来自 `animal` 的内容。第二步 feed-forward 就像每个人回到座位后整理笔记：把刚收来的信息重新加工、筛选、组合，形成更好用的表达。

把这个类比对应到代码：

```python
self.self_attn = self_attn
self.feed_forward = feed_forward
self.sublayer = clones(SublayerConnection(size, dropout), 2)
```

这里的 `feed_forward` 不是在 `EncoderLayer` 里面现场创建的，而是从外面传进来的。它的源头在第 22 章 `make_model()`：

```python
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
```

也就是说，代码先创建了一个前馈网络 `ff`，然后在组装 encoder layer 时把它传进去：

```python
EncoderLayer(d_model, c(attn), c(ff), dropout)
```

这里的 `c(ff)` 是 `copy.deepcopy(ff)` 的简写，意思是复制一份前馈网络给这一层用。这样每一层结构一样，但参数是独立的，不会所有 encoder layer 共用同一组权重。

传进来以后，`__init__` 里的这行：

```python
self.feed_forward = feed_forward
```

就是把外面传进来的前馈网络保存到当前这个 `EncoderLayer` 里面。后面 `forward()` 才能调用：

```python
return self.sublayer[1](x, self.feed_forward)
```

所以链路是：

```text
PositionwiseFeedForward 类
        ↓
make_model() 里创建 ff
        ↓
EncoderLayer(..., c(ff), ...)
        ↓
self.feed_forward = feed_forward
        ↓
forward() 里调用 self.feed_forward
```

`self.self_attn` 是“开会交流”的机器。它让每个 token 根据注意力权重收集其他 token 的信息。

`self.feed_forward` 是“整理笔记”的机器。它不再让不同 token 互相看，而是对每个 token 自己的向量做进一步加工。

`self.sublayer` 里有两个 `SublayerConnection`，因为这一层有两个主要子层：第 0 个包住 self-attention，第 1 个包住 feed-forward。每个 `SublayerConnection` 都会自动做三件事：LayerNorm、Dropout、残差连接。也就是说，每个子层都不是裸跑，而是带着“稳定数值、防止过拟合、保留原信息”的保护壳。

`forward()` 里的第一行是最关键的：

```python
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
```

这行看起来绕，是因为它把 self-attention 塞进了 `SublayerConnection` 这个保护壳里。展开理解，大概等价于：

$$
x_1=x+\text{Dropout}(\text{SelfAttention}(\text{LayerNorm}(x)))
$$

代码里写 `self.self_attn(x, x, x, mask)`，三个 `x` 分别对应 Query、Key、Value。为什么都是 `x`？因为这是 encoder 的 self-attention，也就是同一句话内部自己看自己。每个词都从同一批输入向量里生成 Q、K、V：

$$
Q=xW^Q,\quad K=xW^K,\quad V=xW^V
$$

虽然传进去都是 `x`，但经过三个不同的线性矩阵后，会变成不同角色。Query 像“我想找什么信息”，Key 像“我这里有什么线索”，Value 像“如果你关注我，我能给你什么内容”。

这里的 `lambda x: ...` 是 Python 语法，意思是“临时定义一个没有名字的小函数”。它不是 Transformer 独有的东西，而是 Python 里很常见的匿名函数写法。

最基本格式是：

```python
lambda 参数: 返回值
```

比如：

```python
lambda n: n + 1
```

它等价于：

```python
def 临时函数(n):
    return n + 1
```

区别是：`lambda` 通常适合写很短、只用一次的小函数；`def` 适合写更完整、更复杂、需要反复调用的函数。

放回这里：

```python
lambda x: self.self_attn(x, x, x, mask)
```

它等价于临时写了一个函数：

```python
def 临时自注意力函数(x):
    return self.self_attn(x, x, x, mask)
```

也就是说，这个 `lambda` 接收一个输入 `x`，然后把这个 `x` 同时当成 Query、Key、Value，连同 `mask` 一起送进 self-attention。

为什么这里要这么写？关键在 `SublayerConnection.forward()` 的定义：

```python
def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
```

它要求第二个参数 `sublayer` 是一个“能接收一个 `x` 并返回加工结果的函数”。也就是说，`SublayerConnection` 希望自己可以这样调用：

```python
sublayer(self.norm(x))
```

可是 self-attention 本来的调用方式不是只传一个参数，而是：

```python
self.self_attn(query, key, value, mask)
```

它需要 4 个东西：Query、Key、Value、mask。于是原文用 `lambda` 包了一层，把“需要 4 个参数的 self-attention”改造成“看起来只需要 1 个参数的函数”：

```python
lambda x: self.self_attn(x, x, x, mask)
```

这样 `SublayerConnection` 就可以统一处理所有子层了。它不需要知道里面到底是 self-attention 还是 feed-forward，只需要调用：

```python
sublayer(self.norm(x))
```

这里还有一个容易混淆的小点：`lambda x` 里的这个 `x`，不是随便新造的数据，而是 `SublayerConnection` 里面传进去的 `self.norm(x)`。也就是说实际过程更像：

```python
归一化后的x = self.norm(x)
子层输出 = self.self_attn(归一化后的x, 归一化后的x, 归一化后的x, mask)
最终输出 = 原始x + dropout(子层输出)
```

为什么不能直接写成下面这样？

```python
self.sublayer[0](x, self.self_attn(x, x, x, mask))
```

因为这样会立刻执行 `self.self_attn(...)`，把执行结果传进去，而不是把“这个子层怎么计算”的函数传进去。`SublayerConnection` 需要的是一个函数，因为它要先自己做 `LayerNorm(x)`，再把归一化后的结果交给这个函数。如果你提前把 self-attention 算完，它就没机会在中间插入 LayerNorm 和 Dropout 的统一流程了。

所以这行代码的意思可以翻译成大白话：

```text
请用第 0 个保护壳处理 x。
保护壳内部先归一化 x。
归一化后，把它交给这个临时函数。
这个临时函数会拿归一化后的 x 去做 self-attention。
最后保护壳再做 Dropout 和残差相加。
```

这就是 `lambda` 在这里的作用：把一个参数形式不一样的子层，包装成 `SublayerConnection` 能统一调用的样子。

第二行是：

```python
return self.sublayer[1](x, self.feed_forward)
```

这一步把刚才交流后的 `x` 送进前馈网络，也同样包一层残差连接、LayerNorm 和 Dropout。展开理解大概是：

$$
x_2=x_1+\text{Dropout}(\text{FeedForward}(\text{LayerNorm}(x_1)))
$$

最终输出的 `x_2` 还是同样形状：

$$
[\text{batch},\text{seq\_len},\text{d\_model}]
$$

也就是说，输入有多少句、每句多少个 token、每个 token 向量多长，输出仍然保持这个形状。EncoderLayer 不会把句子长度改掉，也不会直接输出词概率。它做的是把每个 token 的向量变得更懂上下文。

可以用一个极简例子想象。输入第一个 encoder layer 前：

```text
it 的向量：主要知道“我是代词”
animal 的向量：主要知道“我是动物”
street 的向量：主要知道“我是街道”
```

过 self-attention 后：

```text
it 的向量：混入了很多 animal 的信息，也保留了一点 street 等其他词的信息
```

再过 feed-forward 后：

```text
it 的向量：把“我可能指 animal”“animal 是单数名词”“tired 更像描述 animal”这些线索整理得更适合下一层使用
```

这就是为什么 encoder 要堆很多层。第一层可能学比较直接的关系，第二层在第一层整理过的信息上继续推理，越往后，token 向量就越不像“单个词的字面意思”，越像“这个词在整句话里的作用和含义”。

### 12. Decoder：负责一步步生成目标句子

原文位置：`Decoder`；这段主要包含 `class Decoder`。

```python

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

```

`Decoder` 和 `Encoder` 很像，也是把多个 `DecoderLayer` 串起来，最后做 LayerNorm。区别在于 decoder 不只是理解句子，它要一步一步生成目标句子。

翻译时，decoder 生成第 3 个词时，只能看到第 1、2 个已经生成的词，不能偷看第 4 个正确答案。这个限制靠后面的 `subsequent_mask` 实现。

因此 decoder 的每一层比 encoder 更忙：它既要看自己已经生成的目标端内容，又要看 encoder 提供的源句子信息。

### 13. DecoderLayer：目标自注意力、源注意力和前馈网络

原文位置：`Decoder`；这段主要包含 `class DecoderLayer`。

```python

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

```

`DecoderLayer` 是 decoder 里反复堆叠的基本单元。它比 `EncoderLayer` 多一个注意力子层，所以更容易让人绕晕。EncoderLayer 只需要理解源句子内部关系；DecoderLayer 既要整理已经写出的译文，又要回头查源句子，还要避免偷看未来答案。

先用翻译场景理解它。假设源句子是：

```text
je suis étudiant
```

目标句子正在生成：

```text
<s> I am a ...
```

decoder 当前要判断下一个 token 是不是 `student`。这时候它需要三种能力：

1. 看看目标端已经写了什么：`<s> I am a`。
2. 回头看看源句子里有哪些信息：`je suis étudiant`。
3. 把前两步得到的信息加工成更适合下一层使用的向量。

这三种能力正好对应源码里的三个子层：

| 子层 | 源码里对应 | 它在做什么 |
| --- | --- | --- |
| masked self-attention | `self.self_attn` | 目标端内部自己看自己，但不能看未来 |
| source attention | `self.src_attn` | 目标端拿当前状态去查询 encoder 的源句子理解结果 |
| feed-forward | `self.feed_forward` | 每个目标位置各自整理、加工信息 |

先看 `__init__`：

```python
def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout), 3)
```

`size` 通常就是 `d_model`，也就是每个 token 向量的长度。论文 base 模型里常见是 512。整个 DecoderLayer 输入和输出都保持这个长度。

`self.self_attn` 是目标端自注意力。它处理的是 decoder 已经拿到的目标序列，比如训练时的 `<s> I am a`，或者推理时已经生成出来的部分。它的重点不是看源语言，而是先让目标端已知 token 之间互相交流。

`self.src_attn` 是源注意力，也常叫 encoder-decoder attention 或 cross-attention。它的作用是：decoder 当前正在生成目标词，需要去 encoder 的输出里查源句子信息。比如生成 `student` 时，它应该强烈参考源句子里的 `étudiant`。

`self.feed_forward` 是逐位置前馈网络。它不负责跨 token 查信息，而是对每个位置自己的向量做进一步加工。你可以把它理解成“查完资料后，每个位置自己整理笔记”。

这里的 `feed_forward` 同样来自第 22 章的 `make_model()`。源码先创建：

```python
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
```

然后组装 decoder layer 时传入：

```python
DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
```

对应 `DecoderLayer.__init__` 的参数位置是：

```python
def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
```

所以第三个 `c(attn)` 不是前馈网络，第四个 `c(ff)` 才是 `feed_forward`。进入 `__init__` 后，这行：

```python
self.feed_forward = feed_forward
```

把它保存下来，最后在 `forward()` 的第三步使用：

```python
return self.sublayer[2](x, self.feed_forward)
```

也就是说，DecoderLayer 里的前馈网络不是“查源句子”的部分。查源句子的是 `src_attn`；`feed_forward` 是查完源句子以后，对每个目标位置的向量再做一次加工整理。

`self.sublayer = clones(..., 3)` 说明这一层有 3 个带保护壳的子层。为什么是 3？因为 decoder layer 里刚好有三步：目标自注意力、源注意力、前馈网络。每一步外面都套着 `SublayerConnection`，也就是：

$$
x+\text{Dropout}(\text{子层}(\text{LayerNorm}(x)))
$$

这层保护壳的意义是：先归一化让数字稳定，再跑子层，再通过残差连接保留原信息。这样即使某个子层暂时没学好，原来的信息也不会被完全覆盖。

再看 `forward()` 的输入：

```python
def forward(self, x, memory, src_mask, tgt_mask):
```

`x` 是 decoder 当前的目标端表示。训练时，它来自目标句子的右移输入，比如 `<s> I am a`；推理时，它来自已经生成的 token。形状大致是：

$$
x:[\text{batch},\text{target\_len},\text{d\_model}]
$$

`memory` 是 encoder 对源句子的输出。名字叫 memory 很形象：encoder 已经把源句子读完，并把理解结果存成一份“记忆”。形状大致是：

$$
memory:[\text{batch},\text{source\_len},\text{d\_model}]
$$

`src_mask` 用来遮住源句子里的 padding。比如一个 batch 里短句后面补了 `<blank>`，decoder 查源句子时不应该关注这些假 token。

`tgt_mask` 更关键，它既遮住目标端 padding，又遮住未来位置。比如正在训练第 3 个目标词时，模型不能偷看第 4 个、第 5 个正确答案。

第一行：

```python
m = memory
```

这行没有复杂计算，只是给 `memory` 起了一个短名字 `m`，后面写起来更方便。它表示源句子的 encoder 输出。

第二行：

```python
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
```

这是 decoder 的第一步：目标端 masked self-attention。展开看大概是：

$$
x_1=x+\text{Dropout}(\text{MaskedSelfAttention}(\text{LayerNorm}(x)))
$$

这里 `self.self_attn(x, x, x, tgt_mask)` 里三个 `x` 还是 Q、K、V，和 encoder self-attention 类似，因为它也是“自己看自己”。不同点是这里用了 `tgt_mask`，所以目标端每个位置只能看自己和前面的 token，不能看后面的 token。

举例，目标端是：

```text
<s> I am a student
```

训练时模型虽然一次性拿到了整句目标序列，但在预测 `a` 的位置时，它只能看：

```text
<s> I am
```

不能看后面的：

```text
student
```

否则训练时它会作弊：明明要预测 `student`，却已经提前看到答案。这样训练 loss 会好看，但真正生成时就会出问题，因为推理时未来 token 根本不存在。

第三行：

```python
x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
```

这是 decoder 最有特色的一步：source attention。这里 Q、K、V 不再都来自同一个地方：

$$
Q=\text{decoder 当前状态}
$$

$$
K=\text{encoder 源句子记忆}
$$

$$
V=\text{encoder 源句子记忆}
$$

对应源码就是：

```python
self.src_attn(x, m, m, src_mask)
```

第一个 `x` 是 Query，来自 decoder。它代表“我现在想生成目标端某个词，我需要查什么信息？”两个 `m` 分别是 Key 和 Value，来自 encoder 的 `memory`。Key 负责被匹配，Value 负责真正提供内容。

为什么 Key 和 Value 都是 `memory`？因为 decoder 查的是源句子。encoder 已经把 `je suis étudiant` 每个 token 加工成带上下文的向量，这些向量既可以拿来当“索引线索”匹配 Query，也可以拿来当“内容”传给 decoder。

继续看翻译例子。decoder 当前已经有：

```text
<s> I am a
```

它现在要生成下一个词。经过第一步 masked self-attention 后，当前位置大概知道：“英文这边已经写到 I am a，后面很可能需要一个名词。”但这个名词到底是什么，要查源句子。于是 source attention 用当前 decoder 状态去匹配 encoder memory，发现源句子里的 `étudiant` 最相关，于是把 `étudiant` 对应的 Value 信息大量混入当前目标位置。

这一步和第 11 节的 self-attention 最大区别是：

| 类型 | Query 来自哪里 | Key/Value 来自哪里 | 目的 |
| --- | --- | --- | --- |
| encoder self-attention | 源句子 | 源句子 | 源句子内部互相理解 |
| decoder masked self-attention | 目标句子已生成部分 | 目标句子已生成部分 | 目标端内部整理上下文，不能看未来 |
| decoder source attention | decoder 当前状态 | encoder 源句子记忆 | 生成目标词时回查源句子 |

最后一行：

```python
return self.sublayer[2](x, self.feed_forward)
```

这是第三步：feed-forward。展开大概是：

$$
x_3=x_2+\text{Dropout}(\text{FeedForward}(\text{LayerNorm}(x_2)))
$$

它做的不是“再查别的词”，而是对每个目标位置自己的向量做非线性加工。前两步已经把目标端上下文和源句子信息都混进来了，feed-forward 负责把这些信息整理成下一层更好用的表示。

所以，一个 DecoderLayer 的完整流程可以这样记：

```text
目标端先看自己已经写了什么
        ↓
再拿当前状态去查源句子
        ↓
最后每个位置自己整理加工
```

用公式概括就是：

$$
x_1=x+\text{MaskedSelfAttention}(x)
$$

$$
x_2=x_1+\text{SourceAttention}(x_1,memory)
$$

$$
x_3=x_2+\text{FeedForward}(x_2)
$$

实际代码里每一步还包了 LayerNorm 和 Dropout，所以更准确的是带 `SublayerConnection` 的版本。

输出的 `x_3` 形状仍然是：

$$
[\text{batch},\text{target\_len},\text{d\_model}]
$$

也就是说，DecoderLayer 不直接输出单词，也不直接输出概率。它只是把目标端每个位置的向量加工得更懂“我已经写了什么”和“源句子说了什么”。真正把这个向量变成词表概率的，是后面的 `Generator`。

### 14. subsequent_mask：遮住未来答案

原文位置：`Decoder`；这段主要包含 `def subsequent_mask`。

```python

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

```

`subsequent_mask(size)` 生成一个上三角遮罩，用来禁止 decoder 看未来词。`torch.triu(..., diagonal=1)` 会把主对角线右上方的位置标成 1，再通过比较变成布尔 mask。

如果目标句子长度是 4，那么第 1 个位置只能看第 1 个位置，第 2 个位置能看 1、2，第 3 个位置能看 1、2、3。矩阵大概是：

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
$$

这就是自回归生成的规则：预测下一个词时，只允许用过去和现在的信息，不能用未来答案作弊。

### 15. example_mask：把遮罩矩阵画出来

原文位置：`Decoder`；这段主要包含 `def example_mask`。

```python

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)

```

`example_mask()` 只是把上面的 mask 画出来。它用 `pandas` 整理数据，再用 `altair` 画成可视化图。

图里允许看的位置通常会被画成一种颜色，不允许看的位置画成另一种颜色。这个可视化非常适合和 Illustrated Transformer 对照：前文图解告诉你 decoder 有“遮住未来”的注意力，这里展示遮罩矩阵到底长什么样。

## 三、注意力与多头注意力：Transformer 最核心的计算

这一部分对应 Illustrated Transformer 中 Q/K/V、缩放点积注意力、多头注意力、拼接和输出矩阵。前面那篇文章讲“为什么词要互相看”，这里直接给出“词到底怎么算出自己该看谁”。

![缩放点积注意力](https://cdn.jsdelivr.net/gh/13212133871/LLMLeran@main/images/annotated-transformer/03-scaled-dot-product-attention.png)

![多头注意力](https://cdn.jsdelivr.net/gh/13212133871/LLMLeran@main/images/annotated-transformer/04-multi-head-attention.png)

### 16. attention：缩放点积注意力公式落地

原文位置：`Attention`；这段主要包含 `def attention`。

```python

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

```

`attention()` 是整篇 Transformer 里最核心的计算函数。第 11 章、第 13 章一直在说 self-attention、source-attention，真正“怎么算注意力”的地方就在这里。

先看 Q/K/V 在上层通常怎么传入。

从上层调用看，确实常见两类：

| 场景 | 上层写法 | 直觉 |
| --- | --- | --- |
| encoder self-attention | `self.self_attn(x, x, x, mask)` | 源句子自己看自己 |
| decoder masked self-attention | `self.self_attn(x, x, x, tgt_mask)` | 目标句子已生成部分自己看自己，不能看未来 |
| decoder source attention | `self.src_attn(x, m, m, src_mask)` | decoder 当前状态去查 encoder 的源句子记忆 |

但要注意：`attention(query, key, value, ...)` 这个函数收到的通常已经不是最原始的 `x` 或 `m` 了。中间还有第 17 章 `MultiHeadedAttention.forward()` 里的线性投影：

```python
query, key, value = [
    lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    for lin, x in zip(self.linears, (query, key, value))
]
```

也就是说，上层传进来可能都是 `x`，但进入 `attention()` 前，它们已经分别经过不同线性层，变成了真正的 Q、K、V：

$$
Q=xW^Q,\quad K=xW^K,\quad V=xW^V
$$

所以“传三个一样的 `x`”不等于 Q/K/V 完全一样。它只表示 Q/K/V 都从同一批 token 表示里生成；经过不同矩阵后，它们会变成不同角色。

接着看源码第一行：

```python
d_k = query.size(-1)
```

`d_k` 是每个 Query/Key 向量的维度。多头注意力里，如果 `d_model=512`、`h=8`，那么每个头的 `d_k=64`。后面除以 $$\sqrt{d_k}$$ 就用它。

真正的核心是这一行：

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

这行在做：

$$
\text{scores}=\frac{QK^T}{\sqrt{d_k}}
$$

这里做的是矩阵乘法：用 Query 矩阵乘以 Key 的转置，得到一个“每个 query token 对每个 key token 的匹配分数矩阵”。

为什么 `K` 要转置？因为我们想让每个 Query 向量和每个 Key 向量都做一次点积。

假设一句话里只有 2 个 token，每个向量 3 维。Query 矩阵是：

$$
Q=
\begin{bmatrix}
1&0&2\\
0&1&1
\end{bmatrix}
$$

第一行是第 1 个词的 Query，第二行是第 2 个词的 Query。Key 矩阵是：

$$
K=
\begin{bmatrix}
1&1&0\\
0&1&2
\end{bmatrix}
$$

第一行是第 1 个词的 Key，第二行是第 2 个词的 Key。

如果不转置，`Q` 和 `K` 都是 $$2\times3$$，无法直接做矩阵乘法：

$$
(2\times3)\cdot(2\times3)
$$

中间维度 3 和 2 对不上，不能乘。把 `K` 转置后：

$$
K^T=
\begin{bmatrix}
1&0\\
1&1\\
0&2
\end{bmatrix}
$$

它变成 $$3\times2$$，于是：

$$
QK^T=(2\times3)\cdot(3\times2)=(2\times2)
$$

这个 $$2\times2$$ 的结果非常有意义：它表示“2 个 Query 词”分别和“2 个 Key 词”的匹配分。

手算一下：

$$
QK^T=
\begin{bmatrix}
1&0&2\\
0&1&1
\end{bmatrix}
\begin{bmatrix}
1&0\\
1&1\\
0&2
\end{bmatrix}
=
\begin{bmatrix}
1&4\\
1&3
\end{bmatrix}
$$

这个分数矩阵可以读成：

| | 看第 1 个词 | 看第 2 个词 |
| --- | --- | --- |
| 第 1 个词作为 Query | 1 | 4 |
| 第 2 个词作为 Query | 1 | 3 |

也就是说，第 1 个词更想看第 2 个词，因为 4 比 1 大；第 2 个词也更想看第 2 个词，因为 3 比 1 大。

为什么要除以 $$\sqrt{d_k}$$？点积会随着维度变大而变大。维度越高，很多数字相乘再相加，分数容易变得很大。分数太大时，softmax 会变得过于极端，比如接近 `[0.001, 0.999]`，梯度会不稳定。除以 $$\sqrt{d_k}$$ 相当于把分数缩放回更温和的范围。

如果这里 $$d_k=3$$，就除以：

$$
\sqrt{3}
$$

实际代码就是：

```python
/ math.sqrt(d_k)
```

接下来是 mask：

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
```

mask 的意思是：有些位置不允许看。比如 decoder 不能看未来 token，padding 位置也不应该被关注。把这些位置填成 `-1e9`，是为了让它们经过 softmax 后概率几乎变成 0。

比如某一行原始分数是：

$$
[1,\ 4,\ 2]
$$

如果第三个位置不能看，mask 后变成：

$$
[1,\ 4,\ -1000000000]
$$

softmax 后第三个位置的概率几乎就是 0。

然后是：

```python
p_attn = scores.softmax(dim=-1)
```

`dim=-1` 表示对最后一个维度做 softmax。对于分数矩阵来说，就是“每一行单独变成概率”。每一行代表一个 Query 词，所以 softmax 后，每个 Query 词都会得到一组“我该看各个 Key 词多少比例”的概率。

假设缩放后暂时忽略小数，某一行分数是：

$$
[1,\ 4]
$$

softmax 后可能大概变成：

$$
[0.05,\ 0.95]
$$

意思是：这个词 5% 看第 1 个词，95% 看第 2 个词。

如果有 dropout：

```python
if dropout is not None:
    p_attn = dropout(p_attn)
```

这里的 Dropout 是作用在注意力概率上。训练时它可能随机遮掉一部分注意力连接，逼模型不要永远依赖某一个固定词或固定头。比如模型总是只看某个词，Dropout 有时会把这条注意力路径临时削弱，让模型学习备用线索。

最后一行：

```python
return torch.matmul(p_attn, value), p_attn
```

这一步是用注意力概率去加权求和 Value：

$$
\text{输出}=\text{注意力概率}\cdot V
$$

继续用一个简单 Value 矩阵：

$$
V=
\begin{bmatrix}
10&0\\
0&20
\end{bmatrix}
$$

假设注意力概率是：

$$
P=
\begin{bmatrix}
0.05&0.95\\
0.12&0.88
\end{bmatrix}
$$

那么输出是：

$$
PV=
\begin{bmatrix}
0.05&0.95\\
0.12&0.88
\end{bmatrix}
\begin{bmatrix}
10&0\\
0&20
\end{bmatrix}
=
\begin{bmatrix}
0.5&19\\
1.2&17.6
\end{bmatrix}
$$

这就是“信息交换”的数学含义。每一行输出都是 Value 的加权平均。第 1 个词如果 95% 关注第 2 个词，那么它的新表示里就会大量混入第 2 个词的 Value 信息。

把整段流程串起来：

```text
Q 和 K 做矩阵乘法
        ↓
得到每个词看每个词的相关分数
        ↓
除以 sqrt(d_k) 稳定数值
        ↓
mask 去掉不能看的位置
        ↓
softmax 把分数变成注意力概率
        ↓
用概率加权求和 V
        ↓
得到每个 token 的新向量
```

所以第 16 章这段函数回答的是 Transformer 最核心的问题：一个词不是“凭感觉”去看另一个词，而是用 Query 和 Key 的点积算相关性，再用这个相关性去混合 Value。Q/K 决定“看谁”，V 决定“看到了以后拿走什么信息”。

### 17. MultiHeadedAttention：多个注意力视角并行工作

原文位置：`Attention`；这段主要包含 `class MultiHeadedAttention`。

```python

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

```

`MultiHeadedAttention` 对应论文里的 Multi-Head Attention 模块，也就是原论文图 2 右侧那块：先把 Q/K/V 分别线性投影成多个头，每个头各自做一次 scaled dot-product attention，再把多个头拼接起来，最后过一个输出线性层。

论文公式是：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
$$

$$
\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

这段源码就是在实现这两行公式。

先看 `__init__`：

```python
def __init__(self, h, d_model, dropout=0.1):
    ...
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)
```

`h` 是 head 的数量。论文 base 模型通常是 `h=8`。`d_model` 是每个 token 总向量长度，论文 base 模型通常是 512。

`assert d_model % h == 0` 是在检查能不能平均分头。比如：

$$
512 \div 8 = 64
$$

所以每个头处理 64 维。如果 `d_model=510`、`h=8`，就没法平均切成 8 份，代码会直接报错。

`self.d_k = d_model // h` 表示每个头里的 Q/K/V 维度。论文里常写：

$$
d_k=d_v=d_{model}/h
$$

这份代码假设 $$d_v=d_k$$，所以每个头的 Value 维度也等于 `d_k`。

`self.linears = clones(nn.Linear(d_model, d_model), 4)` 非常关键。它复制了 4 个线性层：

| 第几个线性层 | 用途 | 对应论文 |
| --- | --- | --- |
| 第 1 个 | 生成 Query | $$W^Q$$ |
| 第 2 个 | 生成 Key | $$W^K$$ |
| 第 3 个 | 生成 Value | $$W^V$$ |
| 第 4 个 | 多头拼接后的输出投影 | $$W^O$$ |

为什么每个线性层都是 `d_model -> d_model`？因为先保持总维度不变，比如 512 进、512 出；然后再把这个 512 维拆成 8 个头，每个头 64 维。

这和论文图里的关系是：论文图中 Q、K、V 各自进入 Linear 层，拆成多个并行的 attention 头；代码里的前三个 `self.linears` 就是那三个 Linear。论文图最后的 Linear，对应代码里的 `self.linears[-1]`。

再看 `forward()` 的输入：

```python
def forward(self, query, key, value, mask=None):
```

这三个参数来自上层。它们可能是：

| 场景 | query | key | value |
| --- | --- | --- | --- |
| encoder self-attention | `x` | `x` | `x` |
| decoder masked self-attention | `x` | `x` | `x` |
| decoder source attention | `x` | `memory` | `memory` |

不管上层怎么传，进入 `MultiHeadedAttention` 后，都会先分别过线性层，变成真正的 Q/K/V。

mask 这里有一行：

```python
mask = mask.unsqueeze(1)
```

`unsqueeze(1)` 是在第 1 个维度插入一个新维度。这里最容易卡住的是“为什么要插一维”。先看多头注意力里分数 `scores` 的形状。

经过前面的多头拆分后，attention 分数大致是：

$$
[\text{batch},\text{head},\text{query\_len},\text{key\_len}]
$$

如果是 1 个 batch、8 个头、目标端 4 个位置、源端 6 个位置，那么分数形状就是：

$$
[1,8,4,6]
$$

这表示：每个 batch 里，每个 head 都有一张注意力分数表。每张表的行是 query 位置，列是 key 位置。

mask 是“哪些位置允许看，哪些位置不允许看”的规则表。比如 source padding mask 可能只关心 key 位置：

$$
[\text{batch},1,\text{seq\_len}]
$$

这个形状可以理解成：

```text
每个 batch
    一张规则表
        每个 key 位置能不能看
```

如果某个源句子实际只有 4 个 token，后面补了 2 个 `<blank>`，mask 可能像：

$$
[1,\ 1,\ 1,\ 1,\ 0,\ 0]
$$

其中 1 表示可以看，0 表示不能看。问题是：attention 分数里有 head 这一维，但原来的 mask 没有 head 这一维。也就是说，分数表像：

$$
[\text{batch},\text{head},\text{query\_len},\text{key\_len}]
$$

而 mask 像：

$$
[\text{batch},1,\text{key\_len}]
$$

中间少了 head 这个位置。`unsqueeze(1)` 就是在 batch 后面插入一个“head 维度占位”：

$$
[\text{batch},1,1,\text{seq\_len}]
$$

这里两个 1 很重要。第一个 1 对应 head 维度，意思是“这张规则表可以给所有 head 共用”；第二个 1 对应 query 维度，意思是“每个 query 位置都用同一套 key padding 规则”。

PyTorch 的广播可以理解成：如果某一维是 1，而另一个张量对应维度是 8，那么这份数据不用真的复制 8 份，计算时会自动当作 8 份来用。

用 2 个 head 的小例子看更直观。假设 mask 是：

$$
[1,\ 1,\ 0]
$$

意思是第 3 个 key 是 padding，不能看。加维度后可以想象成：

```text
mask:
batch 1
    head 占位 1
        query 占位 1
            [1, 1, 0]
```

当它遇到 2 个 head 的 scores 时，会广播成：

```text
head 1 使用：[1, 1, 0]
head 2 使用：[1, 1, 0]
```

也就是说，head 1 和 head 2 可以学不同的注意力权重，但都不能去看 padding 位置。规则相同，关注方式可以不同。

如果是 decoder 的未来遮罩，mask 可能是一张下三角表：

$$
\begin{bmatrix}
1&0&0\\
1&1&0\\
1&1&1
\end{bmatrix}
$$

加上 head 维度后，同样会给每个 head 用同一张未来遮罩。第 1 个 head 不能偷看未来，第 2 个 head 也不能偷看未来，所有 head 都必须遵守生成规则。

所以“广播到所有 head 上”的意思不是说 8 个头得到一样的注意力结果，而是说 8 个头共享同一套禁止规则。每个头仍然可以算出不同的注意力分布，只是它们都不能关注 padding，也不能在 decoder 里关注未来 token。

接下来：

```python
nbatches = query.size(0)
```

`nbatches` 是 batch 大小，也就是一次送进模型的句子数量。

最核心的一段是：

```python
query, key, value = [
    lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    for lin, x in zip(self.linears, (query, key, value))
]
```

这段看起来很压缩，可以拆成三步。

第一步，`zip(self.linears, (query, key, value))` 会把前三个线性层分别配给 query、key、value：

```text
第 1 个 Linear 处理 query
第 2 个 Linear 处理 key
第 3 个 Linear 处理 value
```

注意 `self.linears` 有 4 个，但这里只 zip 了 3 个输入，所以这里只用前三个。第 4 个留到最后输出投影时用。

第二步，`lin(x)` 做线性投影。假设输入形状是：

$$
[\text{batch},\text{seq\_len},512]
$$

过 `Linear(512,512)` 后形状还是：

$$
[\text{batch},\text{seq\_len},512]
$$

但内容已经变了。query 过的是 Query 线性层，key 过的是 Key 线性层，value 过的是 Value 线性层，所以三者学到不同角色。

第三步，`view(nbatches, -1, self.h, self.d_k)` 把最后的 512 维拆成 8 个头，每个头 64 维：

$$
[\text{batch},\text{seq\_len},512]
\rightarrow
[\text{batch},\text{seq\_len},8,64]
$$

这里的 `-1` 让 PyTorch 自动推断序列长度。也就是说，如果句子有 10 个 token，它就是 10；如果有 30 个 token，它就是 30。

第四步，`.transpose(1, 2)` 交换第 1 维和第 2 维：

$$
[\text{batch},\text{seq\_len},8,64]
\rightarrow
[\text{batch},8,\text{seq\_len},64]
$$

为什么要交换？因为后面的 `attention()` 希望每个 head 可以像独立小批次一样并行计算。把 head 放到前面后，形状就是：

```text
每个 batch
    每个 head
        每个 token
            64 维向量
```

于是 `attention()` 可以一次性对所有 head 做矩阵乘法，不需要 Python 写 8 次循环。

然后调用第 16 章的核心函数：

```python
x, self.attn = attention(
    query, key, value, mask=mask, dropout=self.dropout
)
```

这时的 `query/key/value` 形状大致是：

$$
[\text{batch},h,\text{seq\_len},d_k]
$$

`attention()` 会在每个 head 内部计算：

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

输出 `x` 的形状仍然是：

$$
[\text{batch},h,\text{seq\_len},d_k]
$$

`self.attn` 保存的是注意力权重，后面第 47 到 51 章可视化注意力时会用到。也就是说，代码不只是算出结果，还把“每个头到底看了哪里”存下来，方便画图观察。

注意力算完以后，要把多个头合回一个向量：

```python
x = (
    x.transpose(1, 2)
    .contiguous()
    .view(nbatches, -1, self.h * self.d_k)
)
```

先 `transpose(1, 2)`：

$$
[\text{batch},h,\text{seq\_len},d_k]
\rightarrow
[\text{batch},\text{seq\_len},h,d_k]
$$

也就是把 head 维度放回 token 后面。

再 `.contiguous()`。这是 PyTorch 的内存细节：`transpose` 后张量在内存里的排列可能不是连续的，直接 `view` 可能不安全，所以先让它变成连续内存。

最后 `view(nbatches, -1, self.h * self.d_k)`：

$$
[\text{batch},\text{seq\_len},8,64]
\rightarrow
[\text{batch},\text{seq\_len},512]
$$

这一步就是论文公式里的 `Concat(head_1, ..., head_h)`。把 8 个头的输出拼回 512 维。

接着：

```python
del query
del key
del value
```

这几行不是数学核心，只是删除临时变量引用，帮助释放内存。理解模型原理时可以不用重点看。

最后：

```python
return self.linears[-1](x)
```

`self.linears[-1]` 是第 4 个线性层，也就是论文里的 $$W^O$$。多个头拼接后只是简单放在一起，还没有充分融合。输出线性层会把这些头的信息重新混合，得到下一层要用的统一表示。

可以用“专家小组”来类比多头注意力。一个句子交给 8 个专家同时看：

```text
第 1 个专家：重点看主谓关系
第 2 个专家：重点看代词指代
第 3 个专家：重点看相邻短语
第 4 个专家：重点看长距离依赖
...
```

每个专家都输出一份 64 维意见，8 份意见拼起来变回 512 维。最后的输出线性层像总编辑，把 8 个专家的意见重新整理成一份统一报告。

多个头为什么会学得不一样？因为前三个 Q/K/V 线性层的参数是可学习的，而且每个头对应的是投影后不同的一段子空间。训练时，不同头收到的梯度不同，参数更新路径也不同。久而久之，它们就可能学会关注不同关系。论文和很多后续可视化都观察到，一些头会偏向局部位置，一些头会偏向句法关系，一些头会偏向源目标对齐。

所以第 17 章的核心意义是：第 16 章只告诉我们“一个注意力头怎么算”；第 17 章把这个计算并行复制成多个视角，并把多个视角合成一个输出。这正是论文 Multi-Head Attention 相比单头注意力的关键：不是只让模型用一种方式看句子，而是让它同时拥有多种可学习的观察角度。

## 四、前馈网络、词嵌入和位置编码：让数字真正带上语言含义

注意力负责让词之间交换信息，但每个位置拿到别人信息后，还需要再加工一下；词本身也要先从文字变成向量；Transformer 没有循环结构，还要额外告诉模型词的顺序。这一组代码就是在解决这三件事。

### 18. PositionwiseFeedForward：每个词位置上的非线性加工

原文位置：`Position-wise Feed-Forward Networks`；这段主要包含 `class PositionwiseFeedForward`。

```python

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

```

`Position-wise Feed-Forward Networks` 可以翻译成“逐位置前馈网络”，也可以更口语地理解成“对每个词位置分别使用的前馈小网络”。

这几个词拆开看：

| 英文 | 中文意思 | 在这里的含义 |
| --- | --- | --- |
| Position-wise | 逐位置、按位置分别处理 | 第 1 个 token 用一次，第 2 个 token 用一次，每个位置单独处理 |
| Feed-Forward | 前馈 | 信息从输入一路算到输出，不像注意力那样再去查别的位置 |
| Networks | 网络 | 这里就是两层线性层加 ReLU 的小神经网络 |

它出现的位置很重要：attention 已经让词和词之间交换过信息，feed-forward 接着对每个 token 自己的向量做加工。可以继续沿用前面的“开会”类比：

```text
Self-Attention：大家开会，互相听取信息。
Feed-Forward：开完会后，每个人回到座位，整理自己的笔记。
```

注意力阶段解决的是“我应该看谁、从别人那里拿什么信息”。前馈网络解决的是“我拿到这些信息后，自己该怎么消化、转换、提炼”。

所以它叫 `Positionwise`：同一个前馈网络会作用到句子里的每个位置，但每个位置是各自处理的。第 1 个词不会在 feed-forward 里直接看第 2 个词，第 2 个词也不会在 feed-forward 里直接看第 3 个词。跨词交流已经在 attention 里做过了。

如果输入形状是：

$$
[\text{batch},\text{seq\_len},d_{model}]
$$

比如：

$$
[2,10,512]
$$

意思是 2 个句子、每句 10 个 token、每个 token 512 维。前馈网络会对这 20 个 token 位置分别应用同一套小网络，输出形状仍然是：

$$
[2,10,512]
$$

它不会改变句子长度，也不会改变 batch 大小。

看 `__init__`：

```python
self.w_1 = nn.Linear(d_model, d_ff)
self.w_2 = nn.Linear(d_ff, d_model)
self.dropout = nn.Dropout(dropout)
```

`w_1` 是第一层线性层，把向量从 `d_model` 维扩展到 `d_ff` 维。论文 base 模型里常见：

$$
d_{model}=512,\quad d_{ff}=2048
$$

也就是：

$$
512 \rightarrow 2048
$$

为什么先变宽？可以理解成给每个 token 一个更大的“思考空间”。512 维是主干表示，2048 维像临时展开的草稿纸。模型可以在更高维空间里组合更多特征，比如语法、语义、位置、搭配关系等。

`w_2` 是第二层线性层，再把 2048 维压回 512 维：

$$
2048 \rightarrow 512
$$

为什么要压回去？因为 Transformer 每个子层外面都有残差连接，需要和原来的 `x` 相加。残差连接要求形状一致。输入是 512 维，输出也必须回到 512 维，才能继续传给下一层。

整体公式是：

$$
\text{FFN}(x)=W_2(\text{Dropout}(\text{ReLU}(W_1x+b_1)))+b_2
$$

更接近代码顺序就是：

```python
self.w_2(self.dropout(self.w_1(x).relu()))
```

从里到外看：

1. `self.w_1(x)`：先把每个 token 的向量从 `d_model` 维变到 `d_ff` 维。
2. `.relu()`：把负数截成 0，加入非线性。
3. `self.dropout(...)`：训练时随机遮掉一部分中间特征，防止过度依赖某些维度。
4. `self.w_2(...)`：再把向量从 `d_ff` 维变回 `d_model` 维。

ReLU 很简单：

$$
\text{ReLU}(x)=\max(0,x)
$$

如果输入是：

$$
[-2,\ 0.5,\ 3,\ -1]
$$

ReLU 后就是：

$$
[0,\ 0.5,\ 3,\ 0]
$$

负数变 0，正数保留。它的意义不只是“改数字”，而是给网络加入非线性能力。如果只有线性层：

$$
W_2(W_1x)
$$

那么两层线性层合起来仍然等价于一层线性层：

$$
W_2W_1x
$$

模型表达能力会受限。加上 ReLU 后，中间出现了“开关”：有些特征通过，有些特征被关掉。这样模型才能学更复杂的模式。

举一个简化例子。假设某个 token 是 `it`，经过 self-attention 后，它已经混入了 `animal` 的信息。现在它的向量里可能同时有这些线索：

```text
我是代词
我可能指向 animal
animal 是单数
tired 更像描述有生命的东西
```

前馈网络做的不是再去看 `animal`，而是把这些已经收集到的线索重新组合。它可能学到这样的内部规则：

```text
如果“代词线索”和“animal 线索”和“tired 线索”同时出现，
就增强“it 指代 animal”的表示。
```

当然，模型不会用中文规则写出来，它是通过矩阵权重和 ReLU 开关学到这种组合关系。

还可以用“逐位置”再强调一下。如果句子有 5 个 token：

```text
The animal was too tired
```

feed-forward 会这样工作：

```text
处理 The 的向量
处理 animal 的向量
处理 was 的向量
处理 too 的向量
处理 tired 的向量
```

每个位置用的是同一套 `w_1` 和 `w_2` 参数，但输入向量不同，所以输出也不同。它像同一个“笔记整理模板”发给每个词，每个词根据自己的笔记内容整理出不同结果。

这就是第 18 章和第 16、17 章的分工：

| 模块 | 主要作用 |
| --- | --- |
| Attention / Multi-Head Attention | 让不同 token 之间交换信息 |
| PositionwiseFeedForward | 对每个 token 已经拿到的信息做非线性加工 |

所以一句话概括：注意力负责“交流”，逐位置前馈网络负责“消化”。两者交替堆叠，token 向量才会一层层变得更有上下文理解能力。

### 19. Embeddings：把 token 编号变成向量

原文位置：`Embeddings and Softmax`；这段主要包含 `class Embeddings`。

```python

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

```

`Embeddings` 是文本进入 Transformer 的第一道“数字化入口”。模型不能直接读 `je suis étudiant` 这种文字，它只能处理数字张量。所以在进入 attention 之前，文字要先经历这样一条链路：

```text
原始文本
    ↓
分词 tokenizer
    ↓
token 列表
    ↓
词表 vocab 查编号
    ↓
token id
    ↓
Embedding 查表
    ↓
向量
    ↓
加位置编码
    ↓
进入 encoder / decoder
```

比如原句是：

```text
je suis étudiant
```

分词后可能得到：

```text
["je", "suis", "étudiant"]
```

词表会给每个 token 一个编号。假设：

| token | 编号 |
| --- | --- |
| `je` | 10 |
| `suis` | 25 |
| `étudiant` | 103 |

那么输入句子就会先变成：

$$
[10,\ 25,\ 103]
$$

但是编号本身没有语义。编号 103 不代表“比 25 大 78，所以更重要”，它只是一个身份证号码。模型不能直接拿这些编号做语义计算，所以需要 embedding 把编号查成向量。

核心就是这一行：

```python
self.lut = nn.Embedding(vocab, d_model)
```

`lut` 可以理解成 lookup table，也就是“查找表”。`nn.Embedding(vocab, d_model)` 会创建一张可学习的大表：

$$
[\text{vocab},d_{model}]
$$

如果 `vocab=30000`、`d_model=512`，这张表就是：

$$
[30000,512]
$$

意思是：词表里有 30000 个 token，每个 token 对应一个 512 维向量。第 0 行是编号 0 的 token 向量，第 1 行是编号 1 的 token 向量，第 103 行就是 `étudiant` 的向量。

用一个很小的例子看，假设 `vocab=5`、`d_model=4`，embedding 表可能长这样：

$$
\begin{bmatrix}
0.10&-0.20&0.03&0.50\\
-0.40&0.90&0.12&-0.10\\
0.33&0.07&-0.80&0.22\\
0.01&0.44&0.31&-0.60\\
-0.12&0.25&0.70&0.08
\end{bmatrix}
$$

如果输入 token id 是：

$$
[1,\ 3,\ 4]
$$

`self.lut(x)` 就是去取第 1、3、4 行：

$$
\begin{bmatrix}
-0.40&0.90&0.12&-0.10\\
0.01&0.44&0.31&-0.60\\
-0.12&0.25&0.70&0.08
\end{bmatrix}
$$

这就是“把 token 编号变成向量”。注意，它不是用编号做数学大小比较，而是用编号去查对应的向量行。

回到 `je suis étudiant`，如果简化成 4 维，可能是：

$$
\text{je}\rightarrow[0.12,-0.30,0.88,0.04]
$$

$$
\text{suis}\rightarrow[0.51,0.09,-0.22,0.73]
$$

$$
\text{étudiant}\rightarrow[-0.44,0.61,0.20,-0.18]
$$

这些向量一开始通常是随机初始化的，不是人工提前写好“je 是第一人称”“étudiant 是学生”。训练时，反向传播会更新这张 embedding 表。比如模型发现 `étudiant` 经常对应英语 `student`，它就会逐渐调整相关向量，让后面的 attention 和 generator 更容易利用这个信息。

`forward()` 里是：

```python
return self.lut(x) * math.sqrt(self.d_model)
```

其中 `self.lut(x)` 就是查表。输入 `x` 如果形状是：

$$
[\text{batch},\text{seq\_len}]
$$

比如：

$$
[2,10]
$$

表示 2 个句子、每句 10 个 token id。查表后会变成：

$$
[\text{batch},\text{seq\_len},d_{model}]
$$

比如：

$$
[2,10,512]
$$

这就和后面 attention、feed-forward 所需的形状接上了。第 16、17 章的 Q/K/V，最开始都是从这些 token 向量一路加工来的。

为什么还要乘以：

$$
\sqrt{d_{model}}
$$

这是为了调整 embedding 的数值尺度。下一章会把位置编码直接加到词向量上：

$$
\text{输入向量}=\text{词向量}+\text{位置编码}
$$

如果词向量数值太小，位置编码可能压过词本身的信息；如果词向量数值太大，位置编码又可能太弱。乘以 $$\sqrt{d_{model}}$$ 是原论文和这份代码采用的一种尺度调整方式，让 embedding 和后续位置编码、注意力计算更匹配。

它和其他章节的关系可以这样看：

| 章节 | 和 Embedding 的关系 |
| --- | --- |
| 第 38 章词表 | 先建立 token 到编号的映射 |
| 第 19 章 Embeddings | 用编号查表，得到初始词向量 |
| 第 20 章 PositionalEncoding | 给词向量加上位置信息 |
| 第 16、17 章 Attention | 用这些向量进一步生成 Q/K/V 并交换信息 |
| 第 6 章 Generator | 最后把隐藏向量再映射回词表概率 |

所以 Embedding 是“从离散文字世界进入连续向量世界”的入口；Generator 则是“从连续向量世界回到离散 token 概率”的出口。一个在模型开头，一个在模型结尾，正好相互对应。

### 20. PositionalEncoding：给模型加入顺序感

原文位置：`Positional Encoding`；这段主要包含 `class PositionalEncoding`。

```python

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

```

`PositionalEncoding` 给模型加入位置信息。第 19 章的 Embedding 只告诉模型“这个 token 是什么”，但没有告诉模型“这个 token 在第几个位置”。Transformer 没有 RNN 那种从左到右一个一个读的结构，所以它天然不知道顺序。如果只给词向量，`I love you` 和 `you love I` 在某种程度上会像同一袋词，只是词的集合相同，顺序信息不够明确。

位置编码要解决的问题就是：

```text
词向量：这个词是什么
位置编码：这个词在哪里
两者相加：这个词在这个位置上是什么意思
```

代码里的整体流程是：先在 `__init__` 里提前造好一张位置编码表 `pe`，然后在 `forward()` 里根据当前句子长度切出需要的部分，加到词向量 `x` 上。

先看初始化：

```python
self.dropout = nn.Dropout(p=dropout)
```

位置编码加到词向量后，也会经过 Dropout。意思是训练时随机遮掉一部分位置/词向量混合后的特征，减少模型对某些固定维度的过度依赖。

接着：

```python
pe = torch.zeros(max_len, d_model)
```

这行先创建一张全 0 的表。形状是：

$$
[\text{max\_len},d_{model}]
$$

默认 `max_len=5000`，如果 `d_model=512`，那就是：

$$
[5000,512]
$$

可以理解成：提前准备 5000 个位置，每个位置有一个 512 维的位置向量。第 0 行是第 0 个位置的位置编码，第 1 行是第 1 个位置的位置编码，依此类推。

然后：

```python
position = torch.arange(0, max_len).unsqueeze(1)
```

`torch.arange(0, max_len)` 会生成：

$$
[0,1,2,\ldots,4999]
$$

`unsqueeze(1)` 把它从一行数字变成一列数字：

$$
\begin{bmatrix}
0\\
1\\
2\\
\vdots\\
4999
\end{bmatrix}
$$

形状从：

$$
[5000]
$$

变成：

$$
[5000,1]
$$

为什么要变成列？因为后面要让每个位置 `pos` 和不同维度的频率相乘，形成一张 `[max_len, d_model/2]` 的表。列向量更方便和后面的频率向量做广播计算。

再看：

```python
div_term = torch.exp(
    torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
)
```

这行是在为不同维度准备不同频率。`torch.arange(0, d_model, 2)` 会取偶数维度索引：

$$
[0,2,4,\ldots]
$$

为什么只取偶数？因为后面偶数维用 sin，奇数维用 cos。每一对维度共享一个频率：

```text
第 0 维：sin
第 1 维：cos
第 2 维：sin
第 3 维：cos
...
```

`div_term` 对应公式里这一部分：

$$
\frac{1}{10000^{2i/d_{model}}}
$$

代码没有直接写除法，而是用 `exp(log(...))` 的方式计算，数值上更稳定，也更适合向量化计算。

正弦余弦位置编码公式是：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

代码里的：

```python
position * div_term
```

就对应：

$$
\frac{pos}{10000^{2i/d_{model}}}
$$

接下来填表：

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

`0::2` 表示从第 0 维开始，每隔 2 个取一次，也就是偶数维：

```text
0, 2, 4, 6, ...
```

`1::2` 表示从第 1 维开始，每隔 2 个取一次，也就是奇数维：

```text
1, 3, 5, 7, ...
```

所以这两行的意思是：

```text
偶数维填 sin
奇数维填 cos
```

为什么用 sin 和 cos？直观说，它们像一组不同频率的波。低频维度变化慢，能表达长距离位置变化；高频维度变化快，能表达短距离位置差异。每个位置最终得到一串独特的“波形指纹”。

比如位置 0、1、2 的位置编码不会一样；位置 10 和位置 11 也会有规律地相近，但不是完全相同。这样模型不但能知道“绝对位置”，也更容易通过线性组合捕捉“相对距离”。

然后：

```python
pe = pe.unsqueeze(0)
```

原来 `pe` 的形状是：

$$
[\text{max\_len},d_{model}]
$$

加一维后变成：

$$
[1,\text{max\_len},d_{model}]
$$

为什么前面加一个 1？因为真正输入 `x` 的形状通常是：

$$
[\text{batch},\text{seq\_len},d_{model}]
$$

位置编码本身不需要为 batch 里的每个句子单独存一份。第一个维度设为 1，后面加到 `x` 上时，PyTorch 会把同一张位置编码广播给 batch 里的每个句子。

再看：

```python
self.register_buffer("pe", pe)
```

这行也很重要。`register_buffer` 的意思是：把 `pe` 注册成模型的一部分，但它不是可训练参数。

它和 `nn.Parameter` 不一样：

| 类型 | 会不会训练更新 | 会不会跟着模型保存 | 会不会跟着 `.to(device)` 去 GPU |
| --- | --- | --- | --- |
| `nn.Parameter` | 会 | 会 | 会 |
| `register_buffer` | 不会 | 会 | 会 |
| 普通变量 `self.pe = pe` | 不会 | 不一定按模型状态管理 | 不一定自动处理得好 |

位置编码在这份实现里是固定公式算出来的，不需要训练，所以不应该放进 `model.parameters()` 让优化器更新。但它又是模型运行必需的东西，保存模型、加载模型、搬到 GPU 时都要跟着走，所以适合用 `register_buffer`。

最后看 `forward()`：

```python
def forward(self, x):
    x = x + self.pe[:, : x.size(1)].requires_grad_(False)
    return self.dropout(x)
```

输入 `x` 是第 19 章 embedding 输出的词向量，形状是：

$$
[\text{batch},\text{seq\_len},d_{model}]
$$

例如：

$$
[32,20,512]
$$

表示一个 batch 有 32 个句子，每句当前长度是 20，每个 token 是 512 维。

`x.size(1)` 取的是句子长度，也就是 `seq_len`。如果当前句子长度是 20，那么：

```python
self.pe[:, : x.size(1)]
```

就是从提前准备好的 5000 个位置里，切出前 20 个位置：

$$
[1,5000,512]\rightarrow[1,20,512]
$$

然后和 `x` 相加：

$$
[\text{batch},20,512]+[1,20,512]
$$

第一个维度的 1 会广播到 batch 大小，比如 32。意思是：batch 里的每个句子都用同一套第 0 到第 19 位的位置编码。

特别提到的细节是：

```python
requires_grad_(False)
```

这表示这段位置编码不需要梯度，不参与反向传播更新。也就是说，训练时模型会更新 embedding 表、Q/K/V 矩阵、前馈网络矩阵、Generator 矩阵等参数，但不会通过梯度去修改这张固定正弦余弦位置编码表。

为什么要这样？因为这里采用的是“固定位置编码”，位置编码由公式决定，不是模型学出来的参数。它像一把固定尺子：第 1 格、第 2 格、第 3 格的位置刻度提前画好，训练时不需要把尺子的刻度也改来改去。

如果不写 `requires_grad_(False)`，由于 `pe` 是 buffer，通常也不会作为优化器参数被更新；但这里显式写出来，是在强调：加到 `x` 上的位置编码只是固定信号，不需要对它求梯度。这样语义更清楚，也避免不必要的梯度记录。

最后 `return self.dropout(x)` 会对“词向量 + 位置编码”的结果做 Dropout。也就是说，进入 encoder/decoder 前的输入已经同时包含：

```text
这个 token 是什么
这个 token 在哪里
训练时做一点随机遮挡，提升泛化
```

把第 19 章和第 20 章连起来看，就是：

$$
\text{输入表示}=\text{Embedding(token id)}\cdot\sqrt{d_{model}}+\text{PositionalEncoding}
$$

Embedding 是可学习的，位置编码在这里是固定的。两者相加以后，模型才真正拿到“带顺序感的词向量”。

### 21. example_positional：画出正弦余弦位置编码

原文位置：`Positional Encoding`；这段主要包含 `def example_positional`。

```python

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)

```

`example_positional()` 把位置编码画出来。你会看到不同维度是不同频率的波，有的变化快，有的变化慢。

为什么不用简单的 1、2、3、4 当位置？因为模型内部主要靠线性层、点积、加法来处理向量。正弦余弦编码让相对距离更容易被线性组合捕捉。比如位置相差 1、相差 10、相差 100，会在不同频率维度上留下规律变化。

这和 Illustrated Transformer 里“把位置编码加到词向量上”的图对应。图里是一加，代码里则具体造出了那张位置表。

### 22. make_model：组装完整 Transformer

原文位置：`Full Model`；这段主要包含 `def make_model`。

```python

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

```

`make_model()` 把前面所有零件装成完整 Transformer。它创建多头注意力、前馈网络、位置编码、encoder、decoder、embedding 和 generator。

默认参数 `N=6`、`d_model=512`、`d_ff=2048`、`h=8` 正是论文 base 模型常见配置。`N=6` 表示 6 层 encoder 和 6 层 decoder；`h=8` 表示 8 个注意力头；`d_ff=2048` 表示前馈网络中间层比主向量维度更宽。

这里要特别注意 `feed_forward` 的来源。第 11 章和第 13 章里的 `feed_forward` 都来自这行：

```python
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
```

`ff` 是一个已经创建好的前馈网络对象。后面组装模型时，它被复制后传进 encoder layer 和 decoder layer：

| 代码 | 传进去的是什么 |
| --- | --- |
| `EncoderLayer(d_model, c(attn), c(ff), dropout)` | `c(attn)` 变成 `self_attn`，`c(ff)` 变成 `feed_forward` |
| `DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)` | 第一个 `c(attn)` 变成 `self_attn`，第二个 `c(attn)` 变成 `src_attn`，`c(ff)` 变成 `feed_forward` |

所以源码的装配关系是：

```text
PositionwiseFeedForward(d_model, d_ff, dropout)
        ↓
ff
        ↓
c(ff) 深拷贝一份
        ↓
传进 EncoderLayer / DecoderLayer 的 feed_forward 参数
        ↓
self.feed_forward = feed_forward
        ↓
forward() 里被 SublayerConnection 包起来调用
```

`c(ff)` 里的 `c` 是 `copy.deepcopy`。它的作用是：每一层都拿到同样结构的前馈网络，但参数彼此独立。这样第 1 层、第 2 层、第 6 层都可以学到不同的加工方式。

最后这一段是参数初始化：

```python
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

训练不是从“空白”开始，而是从一组随机小数开始，再通过 loss 和梯度下降不断调整。这里用的 Xavier 初始化，也叫 Glorot 初始化，目的就是给权重一个比较合理的随机起点。

为什么不能随便随机？因为神经网络有很多层，上一层输出会变成下一层输入。如果权重太大，信号一层层传下去可能越来越大，最后数值爆炸；如果权重太小，信号一层层传下去可能越来越接近 0，后面的层几乎收不到有效信息，反向传播的梯度也会变得很弱。

可以用一个很小的线性层理解：

$$
y=xW
$$

如果 `W` 里的数普遍很大，`y` 会被放大；如果 `W` 里的数普遍很小，`y` 会被压小。深层网络里这种放大或压小会反复发生，所以初始化时就要尽量让输入和输出的数值尺度保持平衡。

Xavier 初始化的核心思想是：根据这一层的输入维度和输出维度，自动决定随机数范围，让前向传播和反向传播时的方差尽量稳定。

这里有两个关键词：

| 名称 | 含义 |
| --- | --- |
| `fan_in` | 这一层有多少个输入连接 |
| `fan_out` | 这一层有多少个输出连接 |

比如一个线性层：

```python
nn.Linear(512, 2048)
```

它的输入维度是 512，输出维度是 2048，所以：

$$
fan\_in=512,\quad fan\_out=2048
$$

`xavier_uniform_` 会从一个均匀分布里抽权重：

$$
W\sim U(-a,a)
$$

其中：

$$
a=\sqrt{\frac{6}{fan\_in+fan\_out}}
$$

如果是 `Linear(512, 2048)`，那么：

$$
a=\sqrt{\frac{6}{512+2048}}
=\sqrt{\frac{6}{2560}}
\approx0.048
$$

也就是说，权重大概会从：

$$
[-0.048,\ 0.048]
$$

这个范围里随机抽。输入输出维度越大，单个权重的随机范围通常越小；输入输出维度越小，范围可以相对大一些。这样做是为了让很多输入相加后，输出整体不会太夸张。

`uniform` 的意思是“均匀分布”：区间里的每个数被抽到的概率差不多。`xavier_uniform_` 末尾的下划线 `_` 是 PyTorch 习惯，表示这个函数会原地修改张量，也就是直接把参数 `p` 里的值改掉。

源码里还有一个条件：

```python
if p.dim() > 1:
```

它表示只对维度大于 1 的参数做 Xavier 初始化。通常这类参数是权重矩阵，比如 Q/K/V 的线性层权重、前馈网络权重、Generator 权重等。偏置 bias 通常是一维向量，不走这里的 Xavier 初始化。

把它放回 Transformer 里，Xavier 初始化会给这些矩阵一个更稳的起点：

```text
Embedding 表
Q/K/V 线性层矩阵
多头注意力输出矩阵
前馈网络 w_1 / w_2
Generator 输出矩阵
```

这些参数仍然是随机的，所以模型一开始不会翻译，只是不会从特别糟糕的数值尺度开始。后面训练时，`loss.backward()` 会计算梯度，优化器再一点点把这些随机起点改成真正有用的参数。

一句话概括：Xavier 初始化不是让模型一开始就聪明，而是让模型一开始“数值别太离谱”，这样信号和梯度能比较顺畅地穿过很多层，训练才更容易开始。

## 五、训练前半程：Batch、mask、loss 和参数更新

模型结构搭好以后，还要教它学习。学习不是“告诉模型规则”，而是让模型先猜、算错多少、再用梯度下降一点点改参数。这里的参数包括词嵌入表、Q/K/V 矩阵、前馈网络矩阵、输出层矩阵等，它们都像函数 $$y=ax+b$$ 里的 $$a$$ 和 $$b$$，一开始乱猜，训练时逐步修正。

### 23. inference_test：先用小模型跑通推理

原文位置：`Inference:`；这段主要包含 `def inference_test`、`def run_tests`。

```python

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)

```

`inference_test()` 是一个工程里的“冒烟测试”。它不追求翻译质量，也不是正式评估模型，而是先确认整条推理管线能不能跑通：模型能不能 encode、decode、generator、拼接下一个 token，形状有没有对上，mask 有没有基本生效。

先看：

```python
test_model = make_model(11, 11, 2)
test_model.eval()
```

`make_model(11, 11, 2)` 创建一个很小的 Transformer。两个 `11` 分别表示源语言词表大小和目标语言词表大小都是 11，最后的 `2` 表示 encoder/decoder 只堆 2 层，而不是论文 base 模型的 6 层。这样做是为了测试快一点。

`test_model.eval()` 表示进入推理模式。推理模式下，Dropout 会关闭，LayerNorm 等模块按推理方式运行。也就是说，测试时不会再随机丢特征，输出更稳定。

接着：

```python
src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
src_mask = torch.ones(1, 1, 10)
```

`src` 是一个假的源句子，里面不是文字，而是 token id。形状是：

$$
[1,10]
$$

意思是 1 个句子，长度 10。`src_mask` 全是 1，表示这 10 个位置都是真 token，没有 padding 需要遮掉。

然后：

```python
memory = test_model.encode(src, src_mask)
ys = torch.zeros(1, 1).type_as(src)
```

`memory` 是 encoder 读完源句子后的结果，也就是源句子的上下文表示。`ys` 是 decoder 当前已经生成的目标序列，一开始只有一个 0。这里的 0 可以理解成起始 token 的编号。

核心循环是：

```python
for i in range(9):
```

它会连续预测 9 个 token。每次循环做一件事：根据源句子 `memory` 和当前已经生成的 `ys`，预测下一个 token，再把这个 token 接到 `ys` 后面。

这一段：

```python
out = test_model.decode(
    memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
)
```

把当前已经生成的目标序列 `ys` 送进 decoder。这里用了 `subsequent_mask(ys.size(1))`，意思是 decoder 只能看已经生成出来的位置，不能看未来。

然后：

```python
prob = test_model.generator(out[:, -1])
```

`out` 是 decoder 对所有当前位置的输出。`out[:, -1]` 表示只取最后一个位置，因为推理时我们只关心“下一个 token 是什么”。这也和第 5、6 章呼应：`EncoderDecoder.forward()` 只产出隐藏向量，真正变成词表概率的是 `generator`。

再看：

```python
_, next_word = torch.max(prob, dim=1)
```

`generator` 输出每个候选 token 的分数或 log 概率。`torch.max(..., dim=1)` 取分数最高的 token id。这个策略叫贪心选择：每一步都选当前看起来最可能的词。

最后：

```python
ys = torch.cat(
    [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
)
```

把新预测出的 `next_word` 接到 `ys` 后面。下一轮 decoder 就能看到更长的目标序列。

这段测试跑出来的结果通常没什么语言意义，因为模型还没训练，权重只是随机初始化。但它很有工程价值：如果这段都跑不通，说明模型结构、mask、generator 或张量形状肯定有问题。

`run_tests()` 连续跑 10 次 `inference_test()`，就是多做几次快速检查。正式项目里也常有这种小测试：不验证效果，只验证管道没有断。

### 24. Batch：把样本、padding 和 mask 打包

原文位置：`Batches and Masking`；这段主要包含 `class Batch`。

```python

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

```

`Batch` 是训练时的数据小包。模型不是一次处理一个句子，而是一次处理一批句子，这样 GPU 可以并行计算。可是一个 batch 里的句子长短不同，所以需要 padding 和 mask。

先看：

```python
self.src = src
self.src_mask = (src != pad).unsqueeze(-2)
```

`src` 是源语言 token id 矩阵，形状通常是：

$$
[\text{batch},\text{src\_len}]
$$

`pad=2` 表示编号 2 是 `<blank>`，也就是补齐用的空白 token。假设某个源句子是：

```text
[5, 8, 9, 2, 2]
```

其中后两个 2 是 padding。`src != pad` 会得到：

```text
[True, True, True, False, False]
```

这就是告诉模型：前 3 个位置是真词，后 2 个位置是补齐，不要看。

`.unsqueeze(-2)` 是加一个维度，让 mask 形状更适合 attention 使用。它大致会从：

$$
[\text{batch},\text{src\_len}]
$$

变成：

$$
[\text{batch},1,\text{src\_len}]
$$

这样每个 query 位置都可以复用同一套 source padding 规则。

目标端更复杂：

```python
self.tgt = tgt[:, :-1]
self.tgt_y = tgt[:, 1:]
```

这是训练 decoder 时非常关键的“右移”。假设目标句子是：

```text
[<s>, I, am, student, </s>]
```

训练时 decoder 的输入是：

```text
[<s>, I, am, student]
```

模型要预测的答案是：

```text
[I, am, student, </s>]
```

也就是说，给模型看前面的 token，让它预测后面的 token。代码里的：

```python
tgt[:, :-1]
```

去掉最后一个 token，作为 decoder 输入；而：

```python
tgt[:, 1:]
```

去掉第一个 token，作为正确答案。

接着：

```python
self.tgt_mask = self.make_std_mask(self.tgt, pad)
```

目标端 mask 要同时处理两件事：

1. padding 位置不能看。
2. 未来位置不能看。

所以 `make_std_mask()` 里先做 padding mask：

```python
tgt_mask = (tgt != pad).unsqueeze(-2)
```

再和未来遮罩合并：

```python
tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
```

`&` 是“并且”。只有同时满足“不是 padding”和“不是未来位置”，这个位置才允许 attention 看。

最后：

```python
self.ntokens = (self.tgt_y != pad).data.sum()
```

`ntokens` 统计目标答案里真正有效的 token 数，不包括 padding。训练 loss 常常按有效 token 数归一化，而不是按句子数归一化。因为有的句子很短，有的句子很长，按 token 统计更公平。

所以 `Batch` 不是简单装数据，它把训练需要的几样东西都准备好了：

```text
src：源句子
src_mask：源句子 padding 遮罩
tgt：decoder 输入
tgt_y：decoder 应该预测的答案
tgt_mask：目标端 padding + 未来遮罩
ntokens：有效目标 token 数
```

后面的 `run_epoch()` 就可以直接拿 `Batch` 来训练，不用每次重复写这些整理逻辑。

### 25. TrainState：记录训练进度

原文位置：`Training Loop`；这段主要包含 `class TrainState`。

```python

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

```

`TrainState` 是一个训练进度记录器。它不参与模型计算，也不会改变预测结果，只负责记账。

这几个字段分别是：

| 字段 | 含义 |
| --- | --- |
| `step` | 当前 epoch 里已经处理了多少个 batch |
| `accum_step` | 实际执行了多少次优化器更新 |
| `samples` | 总共看过多少条样本，也就是多少个句子对 |
| `tokens` | 总共处理过多少个有效 token |

为什么既要数 `samples`，又要数 `tokens`？因为翻译任务里句子长短差异很大。两个 batch 都有 32 个句子，但一个 batch 可能总共 300 个 token，另一个可能 1200 个 token。训练成本更接近 token 数，而不是句子数。

`accum_step` 和梯度累积有关。假设显存不够大，不能一次放很大的 batch，就可以先连续处理几个小 batch，把梯度累积起来，再更新一次参数。这样近似模拟更大的 batch。`accum_step` 记录的是“真正更新参数”的次数。

这些统计值常用于日志输出。比如你看到：

```text
Tokens / Sec
Learning Rate
Loss
```

就知道模型训练速度、学习率变化、loss 是否下降。工程上这很重要，因为训练模型不是只写对公式，还要能判断训练是否正常。

### 26. run_epoch：训练循环如何更新参数

原文位置：`Training Loop`；这段主要包含 `def run_epoch`。

```python

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

```

`run_epoch()` 是真正的训练循环。一个 epoch 可以理解成“把训练数据完整过一遍”。这段函数做的事非常工程化：取 batch、前向计算、算 loss、反向传播、更新参数、调学习率、打印日志、统计平均 loss。

先看输入参数：

| 参数 | 含义 |
| --- | --- |
| `data_iter` | 数据迭代器，每次吐出一个 `Batch` |
| `model` | Transformer 模型 |
| `loss_compute` | 负责把模型输出变成 loss，并处理 generator |
| `optimizer` | 根据梯度更新参数 |
| `scheduler` | 调整学习率 |
| `mode` | 当前是训练还是只记录日志 |
| `accum_iter` | 多少个 batch 累积一次梯度 |
| `train_state` | 训练进度记录器 |

开头这些变量：

```python
start = time.time()
total_tokens = 0
total_loss = 0
tokens = 0
n_accum = 0
```

都是为了统计。`total_loss` 和 `total_tokens` 用来算整个 epoch 的平均 loss；`tokens` 和 `start` 用来算阶段性的每秒 token 数；`n_accum` 记录实际更新了多少次参数。

核心循环是：

```python
for i, batch in enumerate(data_iter):
```

`data_iter` 每次给一个 `Batch`，也就是第 24 章整理好的数据包。`i` 是当前 batch 编号。

第一步前向计算：

```python
out = model.forward(
    batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
)
```

这里把源句子、目标输入、源 mask、目标 mask 送进模型。输出 `out` 还不是词概率，而是 decoder 的隐藏向量。后面 `loss_compute` 会调用 `generator` 把它变成词表概率。

第二步算 loss：

```python
loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
```

`batch.tgt_y` 是正确答案，`batch.ntokens` 是有效 token 数。这里返回两个东西：

```text
loss：用于日志统计的数值
loss_node：保留计算图的 loss 张量，可以 backward
```

这两个名字容易混。可以简单理解：`loss` 负责给人看，`loss_node` 负责给 PyTorch 反向传播用。

第三步只有训练模式才会做：

```python
if mode == "train" or mode == "train+log":
```

如果只是验证或测试，就不应该更新参数。训练模式下才执行：

```python
loss_node.backward()
```

这一步根据 loss 反向传播，计算每个参数的梯度。注意，这里还没有真正修改参数，只是把“该怎么改”的建议存到各参数的 `.grad` 里。

然后更新训练状态：

```python
train_state.step += 1
train_state.samples += batch.src.shape[0]
train_state.tokens += batch.ntokens
```

它记录已经处理了几个 batch、多少句子、多少有效 token。

接着是梯度累积：

```python
if i % accum_iter == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    n_accum += 1
    train_state.accum_step += 1
```

如果 `accum_iter=1`，基本就是每个 batch 更新一次参数。如果 `accum_iter=4`，可以理解成处理 4 个 batch 的梯度后，再更新一次参数。这样能在显存有限时模拟更大的 batch。

`optimizer.step()` 才是真正改参数：

$$
\theta \leftarrow \theta-\eta\frac{\partial \text{loss}}{\partial \theta}
$$

`optimizer.zero_grad(set_to_none=True)` 清空梯度，为下一轮重新累计梯度做准备。如果不清空，PyTorch 默认会把新梯度加到旧梯度上，导致更新不符合预期。

然后：

```python
scheduler.step()
```

更新学习率。Transformer 常用 warmup + 衰减的学习率曲线，第 27 章会讲。

后面这些：

```python
total_loss += loss
total_tokens += batch.ntokens
tokens += batch.ntokens
```

是在累计日志数字。`total_loss / total_tokens` 最后会得到平均每个 token 的 loss。

每隔一段时间打印一次：

```python
if i % 40 == 1 and (mode == "train" or mode == "train+log"):
```

日志里有：

```text
当前 batch 编号
参数更新次数
当前 loss
每秒处理多少 token
当前学习率
```

这些信息能帮助判断训练是否健康。比如 loss 一直不降，可能是数据、mask、学习率或模型结构有问题；tokens/sec 太低，可能是数据加载或硬件利用率有问题。

最后：

```python
del loss
del loss_node
```

这是释放临时变量，帮助 PyTorch 更早释放计算图相关内存。深度学习训练很吃显存，这类小细节在工程代码里很常见。

整个 `run_epoch()` 可以概括成：

```text
拿一个 Batch
    ↓
模型前向计算
    ↓
算 loss
    ↓
反向传播得到梯度
    ↓
按设定频率更新参数
    ↓
记录 loss / token 数 / 学习率 / 速度
    ↓
进入下一个 Batch
```

它对应的学习本质仍然是：

$$
\text{预测}\rightarrow\text{计算错误}\rightarrow\text{求梯度}\rightarrow\text{更新参数}
$$

只不过真实训练代码还要处理 batch、mask、日志、显存、学习率、梯度累积这些工程问题。

### 27. rate：Transformer 的学习率曲线

原文位置：`Optimizer`；这段主要包含 `def rate`。

```python

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

```

`rate()` 实现论文里的学习率调度。Transformer 原论文没有一直用固定学习率，而是先 warmup 增大，再按步数衰减。

公式是：

$$
lr=d_{model}^{-0.5}\cdot\min(step^{-0.5}, step\cdot warmup^{-1.5})
$$

为什么要先升后降？训练刚开始，参数很随机，学习率太大容易把模型冲坏，所以先慢慢热身；过了热身期后，再逐渐降低学习率，让模型细致收敛。像学开车，刚上路先慢一点，熟悉后可以加速，快到停车点又要减速。

### 28. example_learning_schedule：把学习率变化画出来

原文位置：`Optimizer`；这段主要包含 `def example_learning_schedule`。

```python

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


example_learning_schedule()

```

`example_learning_schedule()` 把不同配置下的学习率曲线画出来。它不是模型必须组件，而是帮助读者看懂 `rate()` 的形状。

如果只看公式可能抽象，画成曲线就很直观：前面上升，后面下降。不同 `model_size` 和 `warmup` 会改变曲线高度和拐点。

这也提醒你：训练神经网络不仅是搭模型，还要安排“怎么学”。同一个模型，学习率设置不合适，可能完全学不动，或者一开始就发散。

### 29. LabelSmoothing：别让模型过度自信

原文位置：`Label Smoothing`；这段主要包含 `class LabelSmoothing`。

```python

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

```

`LabelSmoothing` 实现标签平滑。普通 one-hot 标签会把正确词概率设为 1，其他词设为 0。标签平滑会把正确词从 1 稍微降一点，把一小部分概率分给其他词。

比如词表 5 个词，正确答案原本是：

$$
[0,0,1,0,0]
$$

平滑后可能接近：

$$
[0.025,0.025,0.9,0.025,0.025]
$$

这不是故意教模型答错，而是防止模型过度自信。翻译里有时不止一个合理答案，模型如果被训练成“只有唯一答案概率必须等于 1”，泛化能力会变差。

### 30. example_label_smoothing：观察平滑后的标签

原文位置：`Label Smoothing`；这段主要包含 `def example_label_smoothing`。

```python

# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


show_example(example_label_smoothing)

```

`example_label_smoothing()` 用一个小例子画出标签平滑后的目标分布。`padding_idx` 对应的位置会被置为 0，因为 padding 不是词，不能参与训练。

这段代码的学习价值在于：loss 不是只能拿硬邦邦的 one-hot 来算。我们可以设计更柔和的目标分布，让模型学会“正确答案最重要，但不是对其他所有可能性都极端否定”。

### 31. loss 可视化：预测分布越离谱惩罚越大

原文位置：`Label Smoothing`；这段主要包含 `def loss`、`def penalization_visualization`。

```python

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


show_example(penalization_visualization)

```

这一段构造不同预测分布，观察标签平滑下 loss 怎么变化。它帮助你理解：模型越自信但越错，惩罚越大；模型合理地把大部分概率放在正确答案上，同时保留一点不确定性，惩罚会更合理。

在翻译中，这很常见。例如德语一个词可能翻成 `big`，也可能根据语境翻成 `large`。标签平滑不会直接告诉模型这些同义词都正确，但它会减少“非标准答案全部为 0”的僵硬感。

### 32. Synthetic Data：用复制任务先验机

原文位置：`Synthetic Data`；这段主要包含 `def data_gen`。

```python

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

```

`data_gen()` 生成一个合成复制任务：输入一串随机数字，目标输出同一串数字。这个任务很简单，但非常适合测试 Transformer 是否能学习基本序列映射。

为什么不一上来就训练真实翻译？因为真实翻译涉及分词、词表、数据下载、GPU、评估指标，变量太多。先让模型学“复制”，如果复制都学不会，说明结构、mask、loss 或训练循环肯定有问题。

这就像造车先在空地上测试刹车和方向盘，再上高速。

### 33. SimpleLossCompute：loss、反向传播和优化器

原文位置：`Loss Computation`；这段主要包含 `class SimpleLossCompute`。

```python

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

```

`SimpleLossCompute` 负责把模型输出变成 loss。它的名字里有 `LossCompute`，意思就是“计算损失”。注意：这份代码里的 `SimpleLossCompute` 本身不直接调用 `optimizer.step()`，真正的反向传播和优化器更新是在第 26 章 `run_epoch()` 里完成的。

先把训练链路放在一起看：

```text
model.forward(...)
    ↓
decoder 隐藏向量 out
    ↓
SimpleLossCompute
    ↓
generator 把 out 变成词表 log 概率
    ↓
criterion 比较预测和正确答案，得到 loss
    ↓
run_epoch 里 loss_node.backward()
    ↓
run_epoch 里 optimizer.step()
```

也就是说，第 33 章负责“算错多少”，第 26 章负责“根据错误反向传播并更新参数”。

先看 `__init__`：

```python
def __init__(self, generator, criterion):
    self.generator = generator
    self.criterion = criterion
```

`generator` 是第 6 章讲过的输出层。decoder 输出的是隐藏向量，还不是词表概率。`generator` 会把隐藏向量映射到词表大小，并输出 log 概率。

`criterion` 是损失函数。前面第 29 章用的是 `LabelSmoothing`，它内部又使用了 `KLDivLoss`。可以把 `criterion` 理解成“评分老师”：模型预测和标准答案越接近，loss 越小；差得越远，loss 越大。

真正被调用的是：

```python
def __call__(self, x, y, norm):
```

Python 里实现了 `__call__` 的对象，可以像函数一样调用。所以在第 26 章里可以写：

```python
loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
```

这里的三个输入是：

| 参数 | 含义 |
| --- | --- |
| `x` | 模型 decoder 输出的隐藏向量 |
| `y` | 正确答案 token id，也就是 `batch.tgt_y` |
| `norm` | 有效 token 数，也就是 `batch.ntokens` |

第一行：

```python
x = self.generator(x)
```

这一步把 decoder 隐藏向量变成词表 log 概率。假设 batch 有 2 句，每句目标长度是 4，词表大小是 11，那么形状大概是：

$$
x:[2,4,d_{model}]
\rightarrow
x:[2,4,11]
$$

`[2,4,11]` 的意思是：2 个句子、每句 4 个位置、每个位置都对 11 个候选 token 给出 log 概率。

接着看最绕的一段：

```python
self.criterion(
    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
)
```

损失函数通常希望输入是二维：

```text
[要预测的 token 总数, 词表大小]
```

而模型输出原来是三维：

```text
[batch, target_len, vocab]
```

所以代码用 `view` 把前两个维度压平。比如：

$$
[2,4,11]\rightarrow[8,11]
$$

这表示一共有 8 个位置要预测，每个位置有 11 个候选 token 的 log 概率。

正确答案 `y` 原来形状是：

$$
[2,4]
$$

也要压平成：

$$
[8]
$$

也就是 8 个正确 token id。这样预测和答案才能一一对应：

```text
第 1 个位置的 11 类概率  对  第 1 个正确 token id
第 2 个位置的 11 类概率  对  第 2 个正确 token id
...
第 8 个位置的 11 类概率  对  第 8 个正确 token id
```

`.contiguous()` 是 PyTorch 的内存细节。前面有些张量可能经过转置或切片，内存不连续，直接 `view` 可能出错。先调用 `.contiguous()`，就是把它整理成连续内存，再安全地改变形状。

然后：

```python
/ norm
```

`norm` 是有效 token 数。loss 先把所有有效 token 的错误加起来，再除以有效 token 数，得到平均每个 token 的错误。这样长句和短句比较公平。

如果不除以 `norm`，长句因为 token 多，loss 天然更大，训练更新会更偏向长句。除以有效 token 数后，loss 更像“平均每个词错多少”。

返回值是：

```python
return sloss.data * norm, sloss
```

这里返回两个东西，很容易混：

| 返回值 | 用途 |
| --- | --- |
| `sloss.data * norm` | 给日志统计用，拿到一个不带计算图的数值 |
| `sloss` | 给反向传播用，保留计算图 |

为什么日志用 `sloss.data * norm`？因为 `sloss` 已经除过 `norm`，如果要统计整个 epoch 的总 loss，先乘回 token 数，后面再除以总 token 数，才能得到全局平均。

为什么还要返回 `sloss`？因为 `sloss` 保留了计算图。第 26 章里：

```python
loss_node.backward()
```

这里的 `loss_node` 就是 `sloss`。PyTorch 会从这个 loss 节点沿着计算图往回走，经过 `criterion`、`generator`、decoder、attention、embedding 等模块，计算每个参数的梯度。

反向传播的意义可以这样理解：loss 告诉模型“这次错了多少”，backward 告诉每个参数“你对这个错误贡献了多少，应该往哪个方向改”。比如正确答案是 `student`，但模型给 `student` 的概率很低，loss 会变大；反向传播会让相关参数朝着“下次提高 student 概率”的方向更新。

优化器的角色在下一步。第 33 章只算出 loss 和梯度入口；第 26 章里的：

```python
optimizer.step()
```

才是真正根据梯度修改参数。比如 Adam 优化器会综合当前梯度、历史梯度均值、历史梯度平方等信息，决定每个参数更新多少。可以把 loss/backward/optimizer 分成三件事：

```text
loss：量错多少
backward：算每个参数该怎么改
optimizer：真正动手改参数
```

所以 `SimpleLossCompute` 是训练闭环里的“评分和算账”部分。没有它，模型只会输出一堆概率，但不知道自己错在哪里；有了 loss，后面的反向传播和优化器才有方向。

### 34. greedy_decode：每一步选概率最大的词

原文位置：`Greedy Decoding`；这段主要包含 `def greedy_decode`。

```python

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

```

`greedy_decode()` 是最简单的生成方法：每一步只选当前概率最大的 token。`greedy` 的中文可以理解成“贪心”，意思是只看眼前这一步最优，不提前考虑未来整体是否最好。

先看函数输入：

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
```

| 参数 | 含义 |
| --- | --- |
| `model` | 已经训练好的 Transformer |
| `src` | 源句子 token id |
| `src_mask` | 源句子 padding mask |
| `max_len` | 最多生成多少个 token |
| `start_symbol` | 起始 token，比如 `<s>` 的编号 |

第一行：

```python
memory = model.encode(src, src_mask)
```

先让 encoder 读完整个源句子，得到 `memory`。`memory` 可以理解成源句子的“理解报告”。后面 decoder 每生成一个 token，都会参考这份 `memory`。

第二行：

```python
ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
```

`ys` 是当前已经生成的目标序列。一开始还没有任何目标词，所以先放一个起始符号。假设 `start_symbol=0`，那么：

$$
ys=[0]
$$

`.type_as(src.data)` 是让 `ys` 的数据类型和 `src` 保持一致。因为 token id 通常是整数类型，模型里的 embedding 查表也需要整数 id。

然后进入循环：

```python
for i in range(max_len - 1):
```

已经有了一个起始 token，所以最多还生成 `max_len - 1` 个 token。

每一步先 decode：

```python
out = model.decode(
    memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
)
```

这里 decoder 看到两类信息：

1. `memory`：源句子的 encoder 输出。
2. `ys`：目前已经生成的目标端 token。

`subsequent_mask(ys.size(1))` 是未来遮罩。虽然推理时 `ys` 本来就只有已经生成的 token，没有未来答案，但 decoder 结构仍然需要这个 mask 来保持和训练时相同的自回归规则。

`out` 是 decoder 输出的隐藏向量，形状大致是：

$$
[\text{batch},\text{current\_target\_len},d_{model}]
$$

接着：

```python
prob = model.generator(out[:, -1])
```

`out[:, -1]` 表示只取最后一个位置。为什么只取最后一个？因为当前生成到 `ys`，我们只需要预测下一个 token。前面位置的预测已经在之前步骤用过了，不需要重复选。

例如当前：

```text
ys = [<s>, I, am]
```

decoder 会输出 `<s>`、`I`、`am` 三个位置的隐藏向量，但我们只关心最后 `am` 这个位置之后应该接什么。所以取：

```python
out[:, -1]
```

`generator` 再把这个最后位置的隐藏向量变成词表上的 log 概率。

然后：

```python
_, next_word = torch.max(prob, dim=1)
```

`torch.max(prob, dim=1)` 会在词表维度上找最大值。返回两个东西：

```text
最大分数
最大分数所在的 token id
```

代码用 `_` 接收最大分数，表示这个值后面不用；`next_word` 是我们真正需要的 token id。

比如词表里：

| token | log 概率 |
| --- | --- |
| `I` | -3.2 |
| `am` | -4.1 |
| `a` | -2.0 |
| `student` | -0.3 |
| `</s>` | -2.8 |

最大的是 `student`，那 `next_word` 就是 `student` 的编号。

接着：

```python
next_word = next_word.data[0]
```

这行把张量里的单个 token id 取出来，方便后面填进新张量。现代 PyTorch 里更常见写法可能是 `.item()`，但原文这里用的是 `.data[0]`。

最后拼接：

```python
ys = torch.cat(
    [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
)
```

这行把新 token 接到已有序列后面。假设：

```text
ys = [<s>, I, am, a]
next_word = student
```

拼接后：

```text
ys = [<s>, I, am, a, student]
```

下一轮循环，decoder 就会拿更长的 `ys` 继续预测下一个 token。

整体流程可以这样记：

```text
先 encode 源句子
    ↓
目标序列从 <s> 开始
    ↓
decode 当前已生成序列
    ↓
只取最后一个位置
    ↓
generator 得到词表概率
    ↓
选概率最大的 token
    ↓
拼到 ys 后面
    ↓
继续下一步
```

贪心解码的优点是简单、快、容易理解。缺点是它只看当前一步最优，不保证整句最优。比如某一步选了当前概率最高的词，后面可能导致句子别扭；另一个当前概率稍低的词，可能让后面整句更自然。

所以真实翻译系统常用 beam search。beam search 不只保留一个候选，而是同时保留多个候选句子，每一步扩展它们，再选整体分数较好的几条继续走。贪心解码像“每个路口只走眼前看起来最近的一条路”；beam search 像“同时看几条可能路线，走一段再比较”。

不过在这篇代码里，`greedy_decode()` 很适合作为教学和小任务测试：它清楚展示了 Transformer 推理时的自回归生成方式，也和第 23 章 `inference_test()` 的流程对应。

### 35. example_simple_model：训练一个最小复制模型

原文位置：`Greedy Decoding`；这段主要包含 `def example_simple_model`。

```python

# Train the simple copy task.


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)

```

`example_simple_model()` 用前面的合成复制数据训练一个小 Transformer。它把模型、criterion、optimizer、scheduler、训练数据、验证数据串起来。

这段很适合当“端到端最小闭环”看：数据进来，模型预测，loss 衡量错误，优化器更新参数，验证集检查效果，最后用 greedy decoding 看模型会不会复制。

如果你刚学代码，不必一次看懂所有行。先抓住闭环：输入、预测、错误、改参数、再预测。深度学习训练反复做的就是这件事。

## 六、真实翻译系统：数据、分词、训练、搜索和可视化

前面的复制任务只是小实验：输入什么就输出什么，主要验证模型结构没接错。真实翻译要面对德语、英语、词表、padding、GPU、多进程训练、模型平均和注意力可视化。这一部分把论文模型放回真实任务里。

![原文结果表](https://cdn.jsdelivr.net/gh/13212133871/LLMLeran@main/images/annotated-transformer/05-results-table.png)

### 36. load_tokenizers：加载德英分词器

原文位置：`Data Loading`；这段主要包含 `def load_tokenizers`。

```python

# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

```

`load_tokenizers()` 加载德语和英语的 spacy 分词器。如果本地没有对应模型，代码会尝试下载。

分词器的任务是把自然语言切成 token。中文里“我喜欢机器学习”可以切成词或字；英文里 `I'm` 可能切成 `I` 和 `'m`；德语有复合词和变格，也需要专门规则。分词质量会影响后面词表和模型输入。

这一步和 Illustrated Transformer 里的“输入词变成 embedding”之间差了一个现实步骤：真实文本要先清洗、切分、编号，才能查 embedding 表。

### 37. tokenize 与 yield_tokens：把文本切成 token

原文位置：`Data Loading`；这段主要包含 `def tokenize`、`def yield_tokens`。

```python

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

```

`tokenize()` 调用分词器把文本变成 token 列表。`yield_tokens()` 则从数据迭代器里不断产出某一列语言的 token，用来构建词表。

比如一条样本可能是 `(德语句子, 英语句子)`。当 `index=0` 时，它产出德语 token；`index=1` 时，它产出英语 token。

你可以把词表构建理解成做一本字典：先把训练集中出现过的 token 全部统计出来，再给每个 token 分配一个整数编号。后面模型看到的就是这些编号。

### 38. build_vocabulary：建立词表和特殊符号

原文位置：`Data Loading`；这段主要包含 `def build_vocabulary`、`def load_vocab`。

```python

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

```

`build_vocabulary()` 和 `load_vocab()` 负责构建或读取词表。特殊符号包括 `<s>` 起始、`</s>` 结束、`<blank>` padding、`<unk>` 未知词。

这些特殊符号很重要。`<s>` 告诉 decoder 从哪里开始生成；`</s>` 告诉它什么时候结束；`<blank>` 用来补齐 batch；`<unk>` 处理词表外的词。

没有这些符号，模型就像拿到一本没有标点和页码的书，不知道句子从哪开始、到哪结束、哪些位置只是空白占位。

### 39. collate_batch：把文本批量变成张量

原文位置：`Iterators`；这段主要包含 `def collate_batch`。

```python

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

```

`collate_batch()` 把 DataLoader 取到的一批原始文本样本变成张量。它会分词、加起止符号、查词表编号、padding，然后把结果堆成矩阵。

举个简化例子，两句英语目标句：

$$
[<s>, I, am, </s>]
$$

$$
[<s>, I, am, a, student, </s>]
$$

为了放在同一个 batch，第一句会补 `<blank>`，变成同样长度。模型计算时靠 mask 忽略这些 `<blank>`。

### 40. create_dataloaders：持续给模型喂 batch

原文位置：`Iterators`；这段主要包含 `def create_dataloaders`。

```python

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

```

`create_dataloaders()` 创建训练集和验证集的 DataLoader。DataLoader 的作用是按 batch 送数据，并调用 `collate_batch()` 把文本整理成模型输入。

训练时不是一次把全部数据塞进模型，而是分批。这样内存压力小，也能让梯度更新更频繁。每个 batch 都像一小套练习题，模型做完、改错，再做下一套。

### 41. train_worker：真实训练主流程

原文位置：`Training the System`；这段主要包含 `def train_worker`。

```python

def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

```

`train_worker()` 是真实训练的核心工作函数，尤其支持多 GPU 分布式训练。它会创建 dataloader、模型、criterion、optimizer、scheduler，然后循环多个 epoch。

对初学者来说，分布式细节可以先不深究。抓主线：每个 worker 负责一部分计算，多张 GPU 协作训练同一个模型。大模型和大数据集训练很慢，多 GPU 是为了加速。

这里也体现了工程代码和论文图解的区别：论文图只画核心结构，真正训练还要处理设备、进程、日志、保存模型等很多工程问题。

### 42. 训练入口与加载模型：单机、多卡和权重文件

原文位置：`Training the System`；这段主要包含 `def train_distributed_model`、`def train_model`、`def load_trained_model`。

```python

def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if is_interactive_notebook():
    model = load_trained_model()

```

这一块提供启动分布式训练、普通训练、加载已训练模型的函数。`train_distributed_model` 用多 GPU，`train_model` 用单进程方式，`load_trained_model` 直接加载权重。

权重文件保存的是训练后的参数，也就是那些 embedding 表、Q/K/V 矩阵、前馈网络矩阵等具体数字。模型结构像空机器，权重像调好位置的零件。只保存结构没有学习成果，只保存权重没有结构也无法运行。

### 43. BPE、搜索和权重共享：进阶系统技巧

原文位置：`Additional Components: BPE, Search, Averaging`；这段主要包含 `if False:`。

```python

if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight

```

`if False:` 里的代码默认不执行，它展示的是一些附加技巧，例如共享 embedding 权重、使用 BPE、beam search、模型平均等。

共享权重的直觉是：源语言和目标语言有时可以共用部分表示，尤其在同一种语言建模或共享子词表时。BPE 则是把词拆成更小的子词单位，缓解生僻词问题。

这一块像原文给读者留的“进阶工具箱”：基础 Transformer 能跑以后，真实系统还会加入这些技巧提升效果。

### 44. average：模型平均让结果更稳

原文位置：`Additional Components: BPE, Search, Averaging`；这段主要包含 `def average`。

```python

def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

```

`average(model, models)` 做模型平均。它把多个模型的参数取平均，放到一个模型里。

为什么平均会有用？训练后期不同 checkpoint 可能都不错，但各自有一点噪声。把它们平均，常常能得到更平滑、更稳定的模型。直观类比是多个老师分别打分，平均分可能比某一个老师的偶然偏差更可靠。

这不是 Transformer 架构本身的一部分，而是训练和推理时常用的工程技巧。

### 45. 结果检查准备：加载数据和模型

原文位置：`Results`；这段主要包含 `# Load data and model for output checks`。

```python

# Load data and model for output checks

```

这一块是结果检查前的准备：加载数据和模型，用于后面验证翻译输出。

如果前面是“搭机器”和“训练机器”，这里就是“拿几句话来看看机器表现如何”。真实项目里，模型是否好不能只看训练 loss，还要看验证集、测试集和具体输出样例。

### 46. check_outputs：看模型实际翻译成什么

原文位置：`Results`；这段主要包含 `def check_outputs`、`def run_model_example`。

```python

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# execute_example(run_model_example)

```

`check_outputs()` 和 `run_model_example()` 会拿训练好的模型做翻译输出检查。它把源句子、目标句子、模型预测放在一起，方便人直接看。

模型评估有两种视角：一种是数字指标，比如 loss、BLEU；另一种是样例观察，比如翻译是否漏词、重复、顺序错。数字能快速比较，样例能帮你发现具体问题。

这和学习 Transformer 原理也一样：只看公式容易抽象，只看图又可能不够精确，公式、代码、样例一起看才更稳。

### 47. attn_map：把注意力矩阵画成人能看的图

原文位置：`Attention Visualization`；这段主要包含 `def mtx2df`、`def attn_map`。

```python

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

```

`mtx2df()` 和 `attn_map()` 把注意力矩阵变成可视化图。注意力矩阵的行可以理解成“当前词”，列可以理解成“被看的词”，格子里的值就是注意力权重。

如果某一行在 `animal` 那一列颜色很深，说明当前词很关注 `animal`。这正好回到 Illustrated Transformer 里 `it` 关注 `animal` 的例子：图片展示现象，代码这里把真实模型里的注意力权重取出来画图。

### 48. visualize_layer：取出不同层和不同头的注意力

原文位置：`Attention Visualization`；这段主要包含 `def get_encoder`、`def get_decoder_self`、`def get_decoder_src`、`def visualize_layer`。

```python

def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))

```

这段定义了从模型里取不同注意力权重的方法：encoder self-attention、decoder self-attention、decoder source-attention。

三者含义不同。encoder self-attention 看源句子内部词与词关系；decoder self-attention 看目标句子已生成部分内部关系；decoder source-attention 看目标词和源句子词之间的对齐关系。

`visualize_layer()` 会把某一层的注意力拿出来画成图。Transformer 有多层多头，所以注意力不是一张图，而是一组图。不同层、不同头可能关注不同语言现象。

### 49. encoder self-attention 可视化

原文位置：`Encoder Self Attention`；这段主要包含 `def viz_encoder_self`。

```python

def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


show_example(viz_encoder_self)

```

`viz_encoder_self()` 可视化 encoder 的自注意力。它展示源语言句子中，每个词在编码阶段关注哪些词。

比如德语句子里，动词可能关注主语，代词可能关注它指代的名词，形容词可能关注它修饰的名词。注意力图不是语法分析器的完美替代，但它能提供一个窗口，让我们看到模型内部某些信息流动方式。

### 50. decoder self-attention 可视化

原文位置：`Decoder Self Attention`；这段主要包含 `def viz_decoder_self`。

```python

def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_self)

```

`viz_decoder_self()` 可视化 decoder 的自注意力。这里一定要记住 mask：decoder 不能看未来，所以注意力图通常呈现下三角可见、右上角被遮住的形态。

如果没有这个遮罩，训练时 decoder 会偷看正确答案后面的词，loss 可能很好看，但真正生成时会崩，因为推理阶段未来词根本不存在。mask 是让训练和推理规则保持一致的关键。

### 51. decoder source-attention 可视化

原文位置：`Decoder Src Attention`；这段主要包含 `def viz_decoder_src`。

```python

def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_src)

```

`viz_decoder_src()` 可视化 decoder 对 encoder 输出的注意力，也就是目标语言生成时看源语言哪里。

这类图常被用来观察翻译对齐。例如生成英文 `student` 时，模型可能强烈关注法语或德语源句里对应“学生”的词。它不是硬规则对齐，而是概率式关注。

到这里，源码和图解就闭环了：前面我们从 `attention()` 公式知道权重怎么计算，从 `MultiHeadedAttention` 知道多头怎么组织，从训练代码知道权重如何学出来，最后从可视化代码看到这些权重在真实翻译样例中长什么样。

## 七、把整篇源码串成一条线

如果把所有代码压缩成一条主线，它其实没有想象中乱：

1. `Embeddings` 把 token id 变成向量。
2. `PositionalEncoding` 把顺序信息加到向量里。
3. `Encoder` 通过多层 self-attention 和 FFN 得到源句子的上下文表示。
4. `Decoder` 用 masked self-attention 看已生成内容，用 source attention 看 encoder 的结果。
5. `Generator` 把 decoder 输出映射到词表概率。
6. `LabelSmoothing` 和 loss 衡量预测离答案有多远。
7. `run_epoch` 通过反向传播和优化器不断更新参数。
8. `greedy_decode` 或更复杂的搜索方法把训练好的模型用于生成。

整篇文章里的矩阵并不是冷冰冰的符号。Q/K/V 矩阵、embedding 表、前馈网络矩阵、输出矩阵，本质都是模型要学习的参数。它们像 $$y=ax+b$$ 里的 $$a$$ 和 $$b$$，只是数量从两个变成了几千万个。训练的核心仍然是：

$$
\text{先预测}\rightarrow\text{算差距}\rightarrow\text{求导数}\rightarrow\text{按梯度修正参数}
$$

和 Illustrated Transformer 对照看，可以形成完整闭环：

| 图解文章解决的问题 | Harvard 源码解决的问题 |
| --- | --- |
| 这个结构为什么这样设计 | 这个结构怎样用 PyTorch 写出来 |
| Q/K/V 的直觉是什么 | Q/K/V 矩阵乘法在哪里发生 |
| 注意力权重怎样理解 | 注意力权重怎样 softmax、mask、乘 Value |
| 多头注意力像多个视角 | 多个 head 怎样 reshape、transpose、concat |
| 训练大概在学什么 | loss、backward、optimizer.step 具体怎么串 |
| 输出词怎么选 | greedy_decode 怎样一步步生成 |

## 八、来源与许可

- 原文：Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- 官方源码：[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)，`the_annotated_transformer.py`
- 源码许可证：MIT License, Copyright (c) 2018 Alexander Rush
- 论文：Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 对照阅读：[`illustrated-transformer-cn.md`](illustrated-transformer-cn.md)

本文档是中文学习笔记，目的是帮助零基础读者把论文图解、数学公式和 PyTorch 源码互相对上。原文代码保留其 MIT License 信息；解释文字为中文重写。
