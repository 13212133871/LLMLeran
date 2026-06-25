# 《nanoGPT 源码中文精讲：从 GPT-2 原理到可运行训练代码》

本文根据 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 官方源码整理，当前参考提交为 `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`。nanoGPT 采用 MIT License。本文目标不是把源码逐字翻译一遍，而是按 `annotated-transformer-cn.md` 的讲法，把每段关键代码和 GPT-2 原理一一对应起来，让没有深度学习工程经验的读者也能看懂它为什么能训练一个小 GPT。

先说结论：nanoGPT 是一个很小但完整的 decoder-only GPT 训练项目。它不是聊天机器人产品，而是一套能训练、加载、采样 GPT 模型的最小工程骨架。它把我们在 `illustrated-gpt2-cn.md` 里讲过的东西落到了代码上：

```text
token embedding      -> self.transformer.wte
position embedding   -> self.transformer.wpe
masked self-attention -> CausalSelfAttention
multi-head attention -> n_head, head_size
MLP                  -> MLP
残差连接              -> Block.forward 里的 x = x + ...
输出词表分数           -> lm_head
下一个 token 预测      -> cross_entropy loss
逐步生成              -> GPT.generate
```

官方 README 也提醒：nanoGPT 现在已经比较老，Karpathy 后来又做了更面向现代聊天模型的 nanochat。但 nanoGPT 仍然非常适合学习，因为它把 GPT 的核心结构压缩在 `model.py` 和 `train.py` 里，代码短、结构清楚、适合精读。

## 1. 先看项目结构：每个文件负责什么

核心文件可以这样理解：

```text
model.py
  定义 GPT 模型本体。包括 LayerNorm、CausalSelfAttention、MLP、Block、GPT。

train.py
  训练主循环。负责读数据、建模型、算 loss、反向传播、优化器更新、保存 checkpoint。

sample.py
  生成文本。负责加载训练好的模型，然后反复调用 model.generate。

data/*/prepare.py
  把原始文本变成 train.bin、val.bin，也就是 token 编号序列。

config/*.py
  训练配置。比如模型多大、batch 多大、学习率多少、训练多少步。
```

一个最小训练流程是：

```text
原始文本
  -> prepare.py 变成 token 编号文件
  -> train.py 读取 batch
  -> model.py 前向计算 logits 和 loss
  -> loss.backward 反向传播
  -> optimizer.step 更新参数
  -> sample.py 加载 checkpoint 生成文本
```

如果把 GPT 比作一个学生：

```text
prepare.py：把教材切成一道道练习题。
model.py：定义学生的大脑结构。
train.py：安排学生刷题、改错、更新能力。
sample.py：考试，让学生根据开头继续写。
```

## 2. 先跑一个最小实战：莎士比亚字符模型

官方建议初学者先跑字符级 Shakespeare，因为它小、快、容易看到结果。

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

这不是训练真正 GPT-2 级别模型，而是训练一个“小 GPT”。它仍然使用同一套核心结构：

```text
embedding
多个 Transformer block
masked self-attention
MLP
lm_head
cross entropy loss
```

不同点是：字符级模型的 token 是一个个字符，例如 `a`、`b`、空格、换行，而 GPT-2 的 token 是 BPE 子词片段。字符模型更小，适合调通流程。

## 3. 数据准备：文字怎么变成数字

`data/shakespeare_char/prepare.py` 的核心代码如下：

```python
chars = sorted(list(set(data)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])
```

这里做的事情很朴素：先统计文本中出现过哪些字符，然后给每个字符分配一个编号。

比如字符表是：

```text
0: 换行
1: 空格
2: !
3: A
4: B
...
```

那么一段文字：

```text
AB A
```

会被编码成：

```text
[3, 4, 1, 3]
```

模型看不懂字符串，只能处理数字编号。后面 `nn.Embedding` 会把这些编号查成向量。

源码接着把数据切成训练集和验证集：

```python
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
```

这和做题一样：大部分题拿来训练，留一部分题用来检查模型有没有真的学会，而不是只背熟训练集。

最后写成二进制文件：

```python
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

`train.bin` 和 `val.bin` 不是神秘格式，它们就是一长串整数。训练时 `train.py` 会从里面随机切出一段段长度为 `block_size` 的序列。

## 4. 训练配置：小模型先看懂超参数

`config/train_shakespeare_char.py` 里有一组小模型配置：

```python
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
```

这些数字对应什么？

```text
dataset：去哪一个 data 子目录读 train.bin 和 val.bin。
batch_size：一次拿多少段文本训练。
block_size：每段文本最长多少 token，也就是上下文长度。
n_layer：堆多少个 Transformer block。
n_head：每层有多少个注意力头。
n_embd：每个 token 向量有多少维。
dropout：训练时随机丢掉一部分通道，防止过拟合。
learning_rate：每次改参数的步子有多大。
max_iters：训练多少轮迭代。
```

如果把模型比作学习机器：

```text
n_layer 决定机器有多少层加工车间。
n_head 决定每层有多少个观察角度。
n_embd 决定每个 token 的信息容量。
block_size 决定模型最多能看多长的前文。
batch_size 决定每次拿多少道题一起练。
```

## 5. GPTConfig：把模型尺寸集中放到一个盒子里

`model.py` 里先定义了配置类：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

`GPTConfig` 就是一张模型施工图。它告诉代码：

```text
词表多大
上下文多长
堆几层
几个注意力头
每个 token 向量多宽
是否使用 dropout
Linear 和 LayerNorm 里是否加 bias
```

GPT-2 small 的经典尺寸接近：

```text
block_size = 1024
vocab_size = 50257
n_layer = 12
n_head = 12
n_embd = 768
```

nanoGPT 默认 `vocab_size=50304`，注释说这是把 GPT-2 的 50257 补到接近 64 的倍数，主要是为了计算效率。可以把它理解成：真实词表有 50257 个 token，但为了让 GPU 处理矩阵更顺手，给表格补了一点空位。

## 6. LayerNorm：让每层输入数值更稳定

源码：

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

LayerNorm 的作用是把每个 token 的向量整理到更稳定的数值范围。深层网络里，如果某一层输出忽大忽小，后面的层就像接到音量忽高忽低的麦克风，很难稳定学习。

`weight` 和 `bias` 是可学习参数。归一化之后，模型还可以自己决定每个维度要不要放大、缩小、平移。

对应公式可以粗略理解为：

$$
\mathrm{LayerNorm}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

这里：

```text
mu：这一组数字的平均值
sigma：这一组数字的标准差
epsilon：防止除以 0 的小数
gamma：源码里的 weight
beta：源码里的 bias
```

## 7. CausalSelfAttention：GPT 的核心注意力层

构造函数核心代码：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
```

这一段和 `illustrated-gpt2-cn.md` 第 22 到 26 章完全对应。

`assert config.n_embd % config.n_head == 0` 是说：总维度必须能平均分给每个注意力头。比如 GPT-2 small：

$$
768 \div 12 = 64
$$

每个头拿 64 维。

`self.c_attn` 是一次性生成 Q、K、V 的大线性层：

$$
768\rightarrow 3\times 768=2304
$$

也就是：

```text
输入 token 向量
  -> 一次线性层
  -> [Query | Key | Value]
```

`self.c_proj` 是多头注意力合并后的输出投影：

```text
多个头拼回 768 维
  -> c_proj
  -> 融合成一个 768 维输出
```

## 8. causal mask 和 Flash Attention：同一个数学，不同实现

源码：

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if not self.flash:
    self.register_buffer(
        "bias",
        torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
    )
```

这里有两个分支。

如果 PyTorch 支持 `scaled_dot_product_attention`，就用 Flash Attention 类似的高效实现。你可以先理解成：PyTorch 内部帮你把注意力计算做得更快、更省显存。

如果不支持，就手动创建一个下三角 mask：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这个 mask 的意义是：第 1 个位置只能看自己，第 2 个位置能看第 1、2 个位置，第 3 个位置能看第 1、2、3 个位置。右上角的未来位置全部禁止。

`register_buffer` 表示把这个 mask 挂到模型里，但它不是训练参数。它会跟着模型保存、移动到 GPU，但不会被梯度下降更新。

## 9. Attention forward：B、T、C 到底是什么

源码开头：

```python
def forward(self, x):
    B, T, C = x.size()
```

这里的 `x` 形状是：

$$
x\in \mathbb{R}^{B\times T\times C}
$$

含义是：

```text
B：batch size，一次训练多少段文本。
T：sequence length，每段文本多少个 token。
C：channel / embedding dim，每个 token 向量多少维。
```

比如小模型可能是：

```text
B = 64
T = 256
C = 384
```

这表示一次训练 64 段文本，每段 256 个 token，每个 token 用 384 个数字表示。

## 10. 一次生成 Q、K、V，再拆成多头

源码：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

先看第一行：

```python
self.c_attn(x)
```

如果 `x` 是：

$$
B\times T\times C
$$

那么 `c_attn` 输出：

$$
B\times T\times 3C
$$

再用 `.split(self.n_embd, dim=2)` 切成三份：

```text
q: B x T x C
k: B x T x C
v: B x T x C
```

然后每一份再拆成多头：

```text
B x T x C
-> B x T x n_head x head_size
-> B x n_head x T x head_size
```

为什么要 `transpose(1, 2)`？因为后面要让每个注意力头独立计算。把 `n_head` 放到前面后，形状变成：

$$
B\times n_{head}\times T\times head\_size
$$

这样可以一次性并行算所有 batch、所有 head 的注意力。

## 11. 手写版注意力：公式如何落地

没有 Flash Attention 时，源码这样写：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
y = att @ v
```

这就是公式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top+M}{\sqrt{d_k}}\right)V
$$

逐行看：

```python
q @ k.transpose(-2, -1)
```

让每个 Query 和每个 Key 做点积。因为 `q` 和 `k` 的形状是：

```text
B x nh x T x hs
```

`k.transpose(-2, -1)` 把最后两维换一下：

```text
B x nh x hs x T
```

矩阵乘法结果是：

```text
B x nh x T x T
```

这张 `T x T` 表就是每个位置看每个位置的分数。

```python
* (1.0 / math.sqrt(k.size(-1)))
```

这是缩放。`k.size(-1)` 是每个头的维度，比如 64。分数除以：

$$
\sqrt{64}=8
$$

避免 softmax 过于极端。

```python
masked_fill(..., float('-inf'))
```

把未来位置改成负无穷。softmax 后这些位置概率就是 0。

```python
y = att @ v
```

用注意力权重去加权求和 Value。注意力分数决定“看谁”，Value 决定“拿什么内容”。

## 12. 重新拼回多个头

源码：

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
return y
```

前面每个头算完后，`y` 的形状是：

```text
B x n_head x T x head_size
```

现在要拼回：

```text
B x T x C
```

因为后面的残差连接和 MLP 都期待每个 token 还是 `C` 维。

`self.c_proj(y)` 对应 GPT-2 图解里的输出投影矩阵。它不是可有可无的小尾巴，而是把多个头的观察结果融合成统一表示。

## 13. MLP：每个 token 自己内部再加工

源码：

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

MLP 的形状是：

$$
C\rightarrow 4C\rightarrow C
$$

如果 `C=768`，就是：

$$
768\rightarrow 3072\rightarrow 768
$$

注意力层负责让 token 之间交流，MLP 负责每个 token 在自己的位置上做更深加工。

可以把一个 Transformer block 想成：

```text
先开会交流：Attention
再回到座位整理笔记：MLP
```

GELU 是非线性激活函数。没有 GELU，两层线性层可以合并成一层线性层，模型表达能力会弱很多。GELU 像一道软门，决定哪些特征更应该通过。

## 14. Block：GPT 的基本积木

源码：

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

这几行非常重要。它就是 GPT block 的核心结构：

```text
LayerNorm -> CausalSelfAttention -> 残差相加
LayerNorm -> MLP                 -> 残差相加
```

为什么是：

```python
x = x + ...
```

这叫残差连接。意思是不要把原来的信息直接覆盖掉，而是在原来的表示上增加一段新加工结果。

可以类比改作文：不是把原文撕掉重写，而是在原文基础上修改、补充、润色。深层网络如果没有残差连接，信息和梯度很容易在多层传播中变弱。

为什么先 `LayerNorm` 再 attention/MLP？这是 pre-norm 写法。它让进入子层的数据更稳定，训练深模型时更不容易炸。

## 15. GPT.__init__：把所有零件装成完整模型

源码：

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

逐个解释：

```text
wte：word token embedding，把 token 编号变成向量。
wpe：word position embedding，把位置编号变成向量。
drop：输入 dropout。
h：很多个 Transformer block。
ln_f：最后的 LayerNorm。
lm_head：把隐藏向量变成词表 logits。
```

如果输入 token 编号是：

```text
[12, 204, 88]
```

`wte` 会把它们查成向量：

```text
3 个 token -> 3 行向量
```

`wpe` 会给位置 0、1、2 也查出位置向量。然后两者相加：

$$
x_t=E_{\text{token}}[id_t]+E_{\text{pos}}[t]
$$

这和 GPT-2 图解中的 token embedding + positional embedding 完全对应。

## 16. 权重绑定：输入词向量和输出词表共用一张表

源码：

```python
self.transformer.wte.weight = self.lm_head.weight
```

这叫 weight tying，权重绑定。

输入时，`wte` 的作用是：

```text
token 编号 -> token 向量
```

输出时，`lm_head` 的作用是：

```text
隐藏向量 -> 词表 logits
```

为什么可以共用？因为词向量表学到的是每个 token 在语义空间里的坐标。输出时也可以用这张坐标表来判断当前隐藏向量更像哪个 token。

数学上可以写成：

$$
logits=hE^\top
$$

其中 $E$ 就是 embedding 矩阵。

## 17. 初始化权重：给训练一个合适起点

源码：

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

训练不是从全 0 开始。全 0 会让很多神经元学到一样的东西，失去分工。这里使用均值为 0、标准差为 0.02 的正态分布随机初始化。

后面还有一段特殊初始化：

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

`c_proj` 是残差路径上的投影层。层数越多，残差叠加越多，如果每一层输出太大，数值可能越来越不稳定。所以这里按层数缩小标准差：

$$
0.02/\sqrt{2L}
$$

其中 $L$ 是层数。

## 18. GPT.forward：训练时 logits 和 loss 怎么来

核心源码：

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
```

`idx` 是 token 编号矩阵：

$$
idx\in \mathbb{R}^{B\times T}
$$

比如：

```text
B=2, T=4
idx =
[
  [10, 20, 30, 40],
  [ 8,  9, 10, 11]
]
```

`wte(idx)` 后变成：

$$
B\times T\times C
$$

`wpe(pos)` 形状是：

$$
T\times C
$$

PyTorch 会自动广播，让每个 batch 都加同一套位置向量。

然后依次穿过所有 block：

```python
for block in self.transformer.h:
    x = block(x)
```

这就是 GPT-2 图解里一层层向上穿过 Transformer stack。

## 19. 训练分支：为什么 logits 要 view 成二维

源码：

```python
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
    )
```

`lm_head(x)` 输出：

$$
B\times T\times vocab\_size
$$

意思是：每个 batch、每个位置，都给整个词表一个分数。

交叉熵函数希望输入形状更像：

```text
所有题目 x 每个题目的候选答案数
```

所以：

```python
logits.view(-1, logits.size(-1))
```

把：

```text
B x T x vocab_size
```

压成：

```text
(B*T) x vocab_size
```

`targets.view(-1)` 把正确答案也压成：

```text
B*T
```

这样每个位置都是一道“根据前文预测下一个 token”的选择题。

## 20. 推理分支：为什么只算最后一个位置的 lm_head

源码：

```python
else:
    logits = self.lm_head(x[:, [-1], :])
    loss = None
```

生成文本时，我们只关心“下一个 token 是什么”。如果输入是：

```text
The robot must
```

模型只需要最后位置 `must` 的隐藏向量来预测下一个 token。前面 `The` 和 `robot` 的输出词表分数已经没用了。

所以推理时只对最后一个位置过 `lm_head`，省一点计算：

```text
x[:, [-1], :] -> B x 1 x C
```

注意这里用 `[-1]` 而不是 `-1`，是为了保留时间维度，让形状仍然是三维。

## 21. generate：模型如何一步步写字

源码：

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

这段就是 GPT-2 自回归生成的代码版。

流程是：

```text
1. 如果上下文太长，只保留最近 block_size 个 token。
2. 前向计算，得到下一个 token 的 logits。
3. 取最后一个位置的 logits。
4. temperature 调整随机性。
5. top_k 只保留概率最高的 k 个候选。
6. softmax 变成概率。
7. multinomial 按概率抽一个 token。
8. 把新 token 拼回 idx，继续下一轮。
```

`temperature` 可以理解成“创造力旋钮”：

```text
temperature < 1：更保守，更倾向高概率词。
temperature = 1：原始概率。
temperature > 1：更随机，更容易出意外。
```

`top_k` 是安全栏。例如 `top_k=200` 表示只从概率最高的 200 个 token 里抽样，避免抽到极低概率的奇怪 token。

## 22. configure_optimizers：为什么有些参数不做 weight decay

源码：

```python
param_dict = {pn: p for pn, p in self.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```

AdamW 是常用优化器。`weight_decay` 可以理解成“别让权重长得太夸张”的正则化。

代码把参数分成两类：

```text
dim >= 2：大矩阵，例如 Linear 权重、Embedding 权重，做 weight decay。
dim < 2：bias、LayerNorm 的 weight/bias，不做 weight decay。
```

为什么 bias 和 LayerNorm 不做？因为它们不是主要的特征映射矩阵，对它们施加权重衰减通常没有必要，甚至可能影响归一化稳定性。

`fused AdamW` 是 CUDA 上更快的实现。如果可用并且设备是 GPU，就启用。

## 23. train.py 的配置覆盖：命令行怎么改参数

源码：

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

nanoGPT 的配置方式很直接：`train.py` 顶部写默认值，然后 `configurator.py` 读取命令行或配置文件来覆盖它们。

所以你可以这样跑：

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

这表示先加载配置文件，再用命令行参数覆盖其中的某些值。

## 24. DDP：多 GPU 训练时每张卡负责一部分

源码片段：

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
```

DDP 是 Distributed Data Parallel，多卡训练用。初学时可以先跳过这块，只记住：

```text
单卡训练：一个进程训练全部。
多卡训练：每张 GPU 各算一部分 batch，然后同步梯度。
```

`master_process` 是主进程。保存 checkpoint、打印日志通常只让主进程做，避免 8 张卡同时写文件。

## 25. get_batch：一段文本如何变成训练题

源码：

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
```

这段非常关键。假设 `block_size=4`，原始 token 序列是：

```text
[10, 20, 30, 40, 50, 60]
```

如果随机起点是 0：

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

模型看到 `x`，要在每个位置预测 `y`：

```text
看到 10          -> 预测 20
看到 10 20       -> 预测 30
看到 10 20 30    -> 预测 40
看到 10 20 30 40 -> 预测 50
```

这就是语言模型训练的本质：输入和目标只差一位。

如果用字符举例：

```text
文本：hello
x：h e l l
y：e l l o
```

模型不断练习“下一个字符是什么”。

## 26. 初始化模型：scratch、resume、gpt2

`train.py` 支持三种初始化：

```python
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
```

三种模式分别是：

```text
scratch：从随机权重开始训练。
resume：从自己的 checkpoint 继续训练。
gpt2*：加载 OpenAI GPT-2 权重再训练或评估。
```

从零训练：

```python
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
```

恢复训练：

```python
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']
```

加载 GPT-2：

```python
model = GPT.from_pretrained(init_from, override_args)
```

这也是 nanoGPT 和 GPT-2 图解的连接点：它的模型结构足够接近 GPT-2，所以能把 GPT-2 权重搬进来。

## 27. estimate_loss：为什么评估时要关掉训练模式

源码：

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

评估时不需要反向传播，所以用 `@torch.no_grad()`，节省显存和计算。

`model.eval()` 会关闭 Dropout 等训练专用行为，让评估更稳定。评估完再 `model.train()` 切回训练模式。

为什么要评估 train 和 val？

```text
train loss 低，val loss 也低：学得不错。
train loss 很低，val loss 高：可能过拟合。
train loss 和 val loss 都高：还没学好，或者模型太小、训练太短。
```

## 28. 学习率调度：warmup + cosine decay

源码：

```python
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

训练一开始，模型权重是随机的。如果学习率一上来太大，容易把参数推飞。所以先 warmup，让学习率从小慢慢升上去。

中后期再用 cosine decay，让学习率逐渐下降。可以理解成：

```text
刚开始：小步试探。
中间：大步学习。
后期：小步精修。
```

## 29. 训练主循环：一次参数更新发生了什么

核心源码：

```python
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这就是训练的心脏。

一轮更新可以拆成：

```text
1. 设置当前学习率。
2. 取一批数据 X, Y。
3. model(X, Y) 得到 logits 和 loss。
4. loss.backward 计算每个参数该怎么改。
5. clip_grad_norm 防止梯度过大。
6. optimizer.step 真正更新参数。
7. zero_grad 清空旧梯度，准备下一轮。
```

`gradient_accumulation_steps` 用来模拟更大的 batch。比如显存只放得下 batch 12，但你想要等效 batch 48，就可以累计 4 次梯度再更新一次参数。

```text
micro step 1：算 loss，累积梯度
micro step 2：算 loss，累积梯度
micro step 3：算 loss，累积梯度
micro step 4：算 loss，累积梯度
optimizer.step：统一更新一次
```

## 30. 混合精度和 GradScaler：更快但要防数值问题

`train.py` 里有：

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

混合精度的意思是：不用所有计算都用 32 位浮点数，有些计算用 16 位也可以，速度更快、显存更省。

但 float16 数值范围更小，梯度可能下溢。`GradScaler` 会把 loss 放大后再反向传播，更新前再缩回来，减少数值问题。

如果是 CPU 初学实验，可以暂时不用理解这些优化，只要知道它们是为了训练更快、更省显存。

## 31. checkpoint：保存的不只是模型权重

源码：

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

checkpoint 像游戏存档。它不只保存模型参数，还保存：

```text
optimizer 状态：AdamW 的动量等信息。
model_args：模型结构参数。
iter_num：训练到第几步。
best_val_loss：目前最好验证损失。
config：当时训练用的配置。
```

如果只保存模型权重，恢复训练时优化器状态丢了，训练节奏会变。完整 checkpoint 能让训练更接近无缝继续。

## 32. sample.py：如何加载模型生成文本

核心源码：

```python
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

`sample.py` 支持两种生成：

```text
从你训练好的 checkpoint 生成。
直接加载 GPT-2 预训练权重生成。
```

加载后：

```python
model.eval()
model.to(device)
```

生成时要切到 eval 模式，因为 dropout 这类训练行为不应该在推理时随机扰动输出。

## 33. prompt 编码：字符串怎么进模型

源码：

```python
if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```

如果你训练的是字符级模型，`meta.pkl` 里有字符表，就用字符级 encode/decode。

如果没有 `meta.pkl`，就默认用 GPT-2 的 BPE tokenizer。也就是：

```text
字符串 -> token id 列表 -> 模型生成更多 token id -> 字符串
```

最后：

```python
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
```

`[None, ...]` 是加一个 batch 维度。原来是一维：

```text
T
```

变成：

```text
1 x T
```

模型永远按 batch 来处理输入，即使只有一个 prompt。

## 34. sample.py 调用 generate

源码：

```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```

这里做了三件事：

```text
torch.no_grad：推理不需要梯度。
ctx：使用混合精度上下文。
model.generate：一步步生成 token。
decode：把 token id 变回文字。
```

`num_samples=10` 表示从同一个开头生成 10 次。因为采样有随机性，每次结果可能不同。

## 35. nanoGPT 和 GPT-2 图解逐项对照

| GPT-2 图解概念 | nanoGPT 源码位置 | 通俗解释 |
|---|---|---|
| token embedding | `wte` | token 编号查表变向量 |
| position embedding | `wpe` | 给 token 加上位置信息 |
| Transformer block 堆叠 | `ModuleList([Block(...)])` | 一层层加工上下文 |
| masked self-attention | `CausalSelfAttention` | 只能看左边，不能看未来 |
| QKV 合并投影 | `c_attn` | 一次性生成 Query、Key、Value |
| 多头注意力 | `view(... n_head ...)` | 把向量切成多个观察角度 |
| causal mask | `torch.tril(...)` 或 `is_causal=True` | 遮住未来 token |
| MLP | `MLP` | 每个 token 自己内部加工 |
| 残差连接 | `x = x + ...` | 保留旧信息，叠加新信息 |
| 最终词表分数 | `lm_head` | 隐藏向量变成每个 token 的分数 |
| 训练损失 | `F.cross_entropy` | 正确下一个 token 概率越低，惩罚越大 |
| 生成 | `generate` | 抽样一个 token，拼回输入，继续 |

## 36. 最容易卡住的几个点

`idx` 不是文字，而是 token id。模型从头到尾都在处理整数和向量。

`targets` 是 `idx` 右移一位。语言模型训练不是人工标注问答，而是从文本自己构造“下一个 token”答案。

`B x T x C` 里的 `C` 是每个 token 的向量宽度，不是类别数。类别数是 `vocab_size`。

`c_attn` 不是注意力结果，它只是生成 Q、K、V 的线性层。

`att` 是注意力权重矩阵，不是最终输出。最终输出是 `att @ v`。

`lm_head` 输出的是 logits，不是概率。概率要经过 softmax。

`loss.backward()` 不会直接更新参数，它只计算梯度。真正更新参数的是 `optimizer.step()`。

`model.eval()` 不是“评估函数”，而是切换模型模式，关闭 Dropout 等训练行为。

## 37. 你可以怎么实战改造

如果只是学习，可以先用字符级 Shakespeare：

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=2000
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

如果要换成自己的文本，最小思路是：

```text
1. 准备一个 input.txt。
2. 参考 data/shakespeare_char/prepare.py，把文本编码成 train.bin 和 val.bin。
3. 写一个 config/my_text.py，设置 dataset、block_size、n_layer、n_head、n_embd。
4. python train.py config/my_text.py
5. python sample.py --out_dir=你的输出目录
```

如果是中文，字符级能跑，但效果有限。更接近 GPT-2 的做法是使用 tokenizer，把中文文本变成 token id。可以参考 nanoGPT 的 `data/shakespeare/prepare.py` 或 OpenWebText 处理方式。

## 38. 从 nanoGPT 到现代 LLM 还差什么

nanoGPT 讲清楚的是 GPT 主干：

```text
decoder-only Transformer
next-token prediction
causal self-attention
训练循环
采样生成
```

现代 LLM 在它基础上还会加入很多工程和训练阶段：

```text
更大的模型和数据
更长上下文
RoPE 等位置编码
RMSNorm / SwiGLU 等结构变化
更高效的注意力和 KV cache
指令微调
偏好对齐
工具调用和聊天模板
分布式训练、FSDP、ZeRO
```

但如果你能看懂 nanoGPT，现代 LLM 的主干就不再神秘。后面的东西大多是在这个主干上做规模化、效率化和对齐训练。

## 39. 一句话复盘 nanoGPT

nanoGPT 把文本变成 token id，把 token id 查成向量，加上位置向量后送入一层层 GPT block。每个 block 先用 causal self-attention 让 token 只和左边上下文交流，再用 MLP 加工每个位置。最后 `lm_head` 把隐藏向量变成词表 logits，训练时用交叉熵惩罚错误预测，生成时从概率分布里抽出下一个 token 并拼回输入。

如果你已经理解 `illustrated-gpt2-cn.md`，那么 nanoGPT 就是把那篇图解变成了可以运行的 PyTorch 代码。

## 来源与许可

本文参考 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 官方仓库，参考源码提交为 `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`。nanoGPT 使用 MIT License。

相关前置阅读：

```text
illustrated-transformer-cn.md
illustrated-gpt2-cn.md
annotated-transformer-cn.md
```
