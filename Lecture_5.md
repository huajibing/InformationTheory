# 第5节：熵率与微分熵

## 〇、引言

上节课我们深入探讨了描述多个随机变量及其相互关系的核心概念，包括联合熵、条件熵、互信息以及Kullback-Leibler散度。本节课将在此基础上进一步拓展，首先回顾这些核心概念，并讨论上次课遗留的问题。随后，我们将引入**熵率 (Entropy Rate)** 的概念，用于描述随机过程平均每个符号所包含的信息量，这对于理解序列数据的压缩极限至关重要。接着，我们将从离散随机变量的熵过渡到连续随机变量，引入**微分熵 (Differential Entropy)** 的概念，探讨其定义、性质以及与离散熵的联系。

---

## 一、上节课内容回顾

### 1. 联合熵 (Joint Entropy) 与 条件熵 (Conditional Entropy)

-   **联合熵 $$H(X,Y)$$**：衡量一对随机变量 $$X$$ 和 $$Y$$ 的总体不确定性。
$$
H(X,Y) = -\sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x,y) \log_2 p(x,y)
$$
-   **条件熵 $$H(X|Y)$$**：在已知随机变量 $$Y$$ 的情况下，随机变量 $$X$$ 的剩余不确定性。其物理意义可以解释为：在平均意义下，如果已知一个随机变量 $$Y$$ 的值，另一个随机变量 $$X$$ 还剩下多少信息量或不确定性。
    $$
    \begin{align*}
    H(X|Y) &= \sum_{y \in \mathcal{Y}} p(y) H(X|Y=y) \\
    &= -\sum_{y \in \mathcal{Y}} p(y) \sum_{x \in \mathcal{X}} p(x|y) \log_2 p(x|y) \\
    &= -\sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x,y) \log_2 p(x|y)
    \end{align*}
    $$
-   **链式法则 (Chain Rule for Entropy)**：
$$
H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
$$
### 2. 互信息 (Mutual Information)

-   互信息 $$I(X;Y)$$ 衡量的是两个随机变量之间共享的信息量，或者说一个随机变量包含的关于另一个随机变量的信息量。
-   定义式：
    $$
    \begin{align*}
    I(X;Y) &= H(X) - H(X|Y) \\
    &= H(Y) - H(Y|X) \\
    &= H(X) + H(Y) - H(X,Y) \\
    &= \sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}
    \end{align*}
    $$
-   **含义**：$$X$$ 中包含 $$Y$$ 的信息量等于 $$Y$$ 中包含 $$X$$ 的信息量。互信息是非负的，即 $$I(X;Y) \ge 0$$，当且仅当 $$X$$ 和 $$Y$$ 相互独立时等号成立。
-   互信息在信息论的后半部分（如带噪信道编码及信道容量）中非常重要。

### 3. KL散度 (Kullback-Leibler Divergence) / 相对熵 (Relative Entropy)

-   KL散度 $$D_{KL}(P||Q)$$ (或 $$KL(P||Q)$$) 用于衡量两个概率分布 $$P$$ 和 $$Q$$ 之间的差异（或“距离”，尽管它不是严格意义上的距离度量）。
-   定义式：
$$
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log_2 \frac{P(x)}{Q(x)}
$$
（约定 $$0 \log_2 \frac{0}{q} = 0$$ 和 $$p \log_2 \frac{p}{0} = \infty$$ 若 $$p>0, q=0$$）
-   **物理含义（从编码角度）**：
    -   如果我们知道信源的真实概率分布 $$P$$，那么最短的平均码长是其熵 $$H(P)$$。
    -   如果我们不知道真实分布 $$P$$，而是使用一个估计的或近似的分布 $$Q$$ 来设计编码（例如，码长 $$l(x) = \log_2 \frac{1}{Q(x)}$$），那么编码的实际平均码长将是 $$\sum_{x} P(x) \log_2 \frac{1}{Q(x)}$$。
    -   这个由于使用错误分布 $$Q$$ 而导致的平均码长相对于使用真实分布 $$P$$ 时的最优平均码长的增量 (gap) 就是 $$D_{KL}(P||Q)$$：
$$
D_{KL}(P||Q) = \left( \sum_{x} P(x) \log_2 \frac{1}{Q(x)} \right) - H(P)
$$
-   **重要性质**：
    -   **非负性 (Gibbs' Inequality)**：$$D_{KL}(P||Q) \ge 0$$，当且仅当 $$P=Q$$ 时等号成立。
    -   **非对称性**：KL散度对于 $$P$$ 和 $$Q$$ **不是对称的**，即一般情况下 $$D_{KL}(P||Q) \neq D_{KL}(Q||P)$$。
-   **与互信息的关系**：$$I(X;Y) = D_{KL}(p(x,y) || p(x)p(y))$$。即互信息是联合分布与边缘分布乘积（独立假设下的分布）之间的KL散度。

---

## 二、上次课遗留问题与讨论

### 1. KL散度与L1范数的关系

-   既然KL散度衡量两个概率分布 $$P$$ 和 $$Q$$ 的差异，我们可以将 $$P=(P_1, \dots, P_d)$$ 和 $$Q=(Q_1, \dots, Q_d)$$ 看作两个向量。衡量向量差异的方式有很多，例如L1范数。
-   **L1范数定义**：
$$
||P-Q||_1 = \sum_{i=1}^d |P_i - Q_i|
$$
-   **问题**：$$D_{KL}(P||Q)$$ 和 $$||P-Q||_1$$ 之间有什么关系？
-   **解答与补充 (Pinsker's Inequality)**：
    Pinsker不等式给出了KL散度和总变差距离（Total Variation Distance, TVD）之间的关系。总变差距离定义为 $$TV(P,Q) = \frac{1}{2} \sum_i |P_i - Q_i| = \frac{1}{2} ||P-Q||_1$$。
    Pinsker不等式的一个常见形式为：
$$
TV(P,Q) \le \sqrt{\frac{1}{2 \ln 2} D_{KL}(P||Q)}
$$
或者等价地：
$$
D_{KL}(P||Q) \ge 2 \ln 2 \cdot (TV(P,Q))^2 = \frac{\ln 2}{2} ||P-Q||_1^2
$$
这意味着如果KL散度很小，那么L1范数（以及总变差距离）也很小。然而，反之不一定成立，特别是当某个 $$Q_i=0$$ 而 $$P_i > 0$$ 时，KL散度为无穷大，但L1范数可能仍然有限。KL散度对分布尾部的差异或 $$Q_i \approx 0$$ 的情况更为敏感。

### 2. 学生讨论与Reduction思路 (关于KL散度的性质)

-   一位同学提出可以将一般的概率分布 $$P=(P_1, ..., P_d)$$ 和 $$Q=(Q_1, ..., Q_d)$$ 通过某种方式"reduce"（聚合或粗化）成更简单的分布（如伯努利分布的推广形式 $$\tilde{P}$$ 和 $$\tilde{Q}$$）来研究。
-   例如，可以将 $$P$$ 和 $$Q$$ 的分量根据 $$P_i \ge Q_i$$ 或 $$P_i < Q_i$$ 分成两部分求和，构造新的二元分布。
    -   教师板书的Reduction示例 (以L1范数为例，并指出其推导过程中的笔误和修正)：
$$
||P-Q||_1 = \sum_{i, P_i \ge Q_i} (P_i - Q_i) + \sum_{i, P_i < Q_i} (Q_i - P_i)
$$
由于 $$\sum P_i = 1$$ 和 $$\sum Q_i = 1$$，所以 $$\sum (P_i - Q_i) = 0$$。
因此，$$\sum_{i, P_i \ge Q_i} (P_i - Q_i) = -\sum_{i, P_i < Q_i} (P_i - Q_i) = \sum_{i, P_i < Q_i} (Q_i - P_i)$$。
所以 $$||P-Q||_1 = 2 \sum_{i, P_i \ge Q_i} (P_i - Q_i)$$。
学生提出的 $$\tilde{P}$$ 和 $$\tilde{Q}$$ 构造意图可能是：
令 $$S_+ = \{i | P_i \ge Q_i\}$$，$$S_- = \{i | P_i < Q_i\}$$。
$$\tilde{P}_1 = \sum_{i \in S_+} P_i, \tilde{P}_2 = \sum_{i \in S_-} P_i$$ (构成一个二元分布 $$(\tilde{P}_1, \tilde{P}_2)$$ )
$$\tilde{Q}_1 = \sum_{i \in S_+} Q_i, \tilde{Q}_2 = \sum_{i \in S_-} Q_i$$ (构成一个二元分布 $$(\tilde{Q}_1, \tilde{Q}_2)$$ )
>        *(助教笔记中指出，教师似乎意在引导学生思考这种 "lumping" 操作对KL散度的影响。)*
-   **教师提问**：在这样的reduction（数据处理或聚合）下，$$D_{KL}(P||Q)$$ 会如何变化？
-   **结论**：会变小或不变。这与KL散度的**数据处理不等式 (Data Processing Inequality for KL Divergence)** 有关。如果我们将原始随机变量 $$X \sim P$$ (或 $$Q$$) 通过一个确定的函数（或随机映射，即一个信道）映射到新的随机变量 $$Y = g(X)$$，得到新的分布 $$P'$$ 和 $$Q'$$，那么 $$D_{KL}(P'||Q') \le D_{KL}(P||Q)$$。聚合操作是一种数据处理。
-   这也与KL散度的**凸性 (Convexity of KL Divergence)** 有关。$$D_{KL}(P||Q)$$ 是关于 $$(P,Q)$$ 对的联合凸函数。

---

## 三、熵率 (Entropy Rate)
> (`L4. Entropy Rate`)

### 1. 引入动机：编码效率与序列编码

#### 1. 单符号编码的局限性回顾
-   对于一个随机变量 $$X$$ 服从概率分布 $$P = (0.01, 0.99)$$。
-   其熵 $$H(X) = -0.01 \log_2 0.01 - 0.99 \log_2 0.99 \approx 0.0808 \text{ bit}$$。
-   使用Huffman编码，其最小平均码长 (Min. Average Code Length) 为 $$1 \text{ bit}$$ (例如，0.01概率的符号编码为'0'，0.99概率的符号编码为'1')。
-   **问题**：实际最优编码的平均码长 ($$1$$ bit/symbol) 与理论下界（熵，约$$0.08$$ bit/symbol）相差较大 (约 $$1 / 0.08 \approx 12.5$$ 倍)。这引出了思考：如何提高编码效率，使平均码长更接近熵？

#### 2. 块编码提高效率 (针对i.i.d.信源)
-   考虑一个独立同分布 (i.i.d. - independent and identically distributed) 的信源，产生序列 $$X_1, X_2, \ldots, X_t, \ldots$$，其中每个 $$X_i \sim P$$。
-   将长度为 $$T$$ 的符号序列 $$(X_1, X_2, \ldots, X_T)$$ 作为一个整体进行编码 (块编码)。
-   对于这个长度为 $$T$$ 的块，其熵为 $$H(X_1, X_2, \ldots, X_T)$$。
-   由于是i.i.d.信源，所以 $$H(X_1, X_2, \ldots, X_T) = \sum_{i=1}^{T} H(X_i) = T \cdot H(X)$$。
-   根据信源编码定理，该数据块的最小平均码长 $$L_{block}$$ 满足：
$$
H(X_1, \ldots, X_T) \le L_{block} < H(X_1, \ldots, X_T) + 1
$$
    即：
$$
T \cdot H(X) \le L_{block} < T \cdot H(X) + 1
$$
-   **平均到每个符号的码长** $$L_{per\_symbol} = \frac{L_{block}}{T}$$：
$$
H(X) \le L_{per\_symbol} < H(X) + \frac{1}{T}
$$
-   当块长度 $$T \to \infty$$ 时，$$L_{per\_symbol} \to H(X)$$。
>    教师板书：$$\frac{T \cdot H(X) + 1}{T} \xrightarrow{T \to \infty} H(X)$$
-   **结论**：通过对越来越长的i.i.d.符号序列进行编码，可以使得单位符号的平均码长无限逼近单个符号的熵 $$H(X)$$。
-   比较编码效率时，应该比较其平均码长的**比率 (ratio)**，而不是绝对差值，尤其是在考虑无限次传输时。
    `Ratio` $$= \frac{L_{block}}{H(X_1, \ldots, X_T)} < \frac{T \cdot H(X) + 1}{T \cdot H(X)} = 1 + \frac{1}{T \cdot H(X)}$$
    当 $$T$$ 很大时，比率趋近于1 (教师板书为 $$1 + O(\frac{1}{T})$$)。

### 2. 随机过程与熵率定义

#### 1. 随机信源 (Random Source / Stochastic Process)：
-   一个真实的信源通常是一个**随机过程 (stochastic process)**，记作 $$\mathcal{X}$$，它产生一个随机变量序列：
    $$X_1, X_2, \ldots, X_t, \ldots, X_{t+1}, \ldots$$ (或者 $$(X_t)_{t \ge 1}$$)
-   这个序列中的随机变量之间**不一定独立**，也不一定**同分布**。
-   对于任意长度为 $$T$$ 的序列 $$(X_1, X_2, \ldots, X_T)$$，它们具有一个**联合概率分布 (joint probability distribution)**。

#### 2. 熵率的定义：
熵率是衡量一个随机过程平均每个符号所携带的信息量（或不确定性）。
-   **定义1 (基于联合熵的平均)**：
$$
\mathcal{H}(\mathcal{X}) = \lim_{T \to \infty} \frac{1}{T} H(X_1, X_2, \ldots, X_T)
$$
    物理意义：随机过程平均每个符号的熵。
-   **定义2 (基于条件熵的极限)**：
$$
\mathcal{H}'(\mathcal{X}) = \lim_{T \to \infty} H(X_T | X_1, X_2, \ldots, X_{T-1})
$$
    物理意义：当已知过去所有历史信息时，下一个符号的条件熵的极限。代表了在知道很长历史之后，新符号带来的“新信息”或“不确定性”。
-   **两个定义的等价性**：
    如果随机过程 $$\mathcal{X}$$ 是**平稳的 (stationary)**，则上述两个极限存在且相等：
$$
\mathcal{H}(\mathcal{X}) = \mathcal{H}'(\mathcal{X})
$$
>    *(教师提及这类似于微积分中的Cesaro均值定理，但未展开。Cesaro均值定理指出，如果一个序列 $$a_n \to L$$，则其算术平均 $$(a_1 + \dots + a_n)/n \to L$$。这里，$$H(X_T | X_1, \ldots, X_{T-1})$$ 对应 $$a_T$$，而 $$\frac{1}{T} H(X_1, \ldots, X_T) = \frac{1}{T} \sum_{k=1}^T H(X_k | X_1, \ldots, X_{k-1})$$ （利用链式法则），所以如果条件熵序列收敛，其均值也收敛到同一极限。)*
-   从信息论角度看，第二种定义更能体现“率”的含义。

### 3. 熵作为下确界的意义回顾

-   之前推导熵 $$H(X) = -\sum P_i \log_2 P_i$$ （板书为 $$\sum P_i \log_2 \frac{1}{P_i}$$）作为最短平均码长的下界时，是通过解决一个带约束（Kraft不等式）的优化问题得到的，且特意**忽略了码长必须为整数**的约束。因此得到的理想码长 $$\log_2 \frac{1}{P_i}$$ 可能不是整数。
-   实际编码（如Huffman编码）码长必须是整数，因此实际最优平均码长 $$L^*$$ 满足 $$H(X) \le L^* < H(X) + 1$$。
-   **在什么意义下，熵 $$H(X)$$ 是平均码长的下确界 (tight lower bound / infimum) 呢？** 这正是引入熵率和考虑对符号序列进行编码的原因。通过对越来越长的符号序列进行编码，单位符号的平均码长可以无限逼近该信源的熵率（对于i.i.d.信源，熵率即为单个符号的熵 $$H(X)$$）。

---

## IV. 微分熵 (Differential Entropy)
> (`L5. Differential Entropy`)

### 1. 从离散熵到微分熵的过渡与问题

#### 1. 离散化连续随机变量及其熵的极限行为
-   到目前为止，所有关于熵和编码的讨论都是基于**离散随机变量**。
-   考虑一个**连续随机变量** $$X$$（例如服从 $$[0,1]$$ 上的均匀分布 $$X \sim U[0,1]$$）。
-   尝试将其离散化：将 $$X$$ 的取值范围（例如 $$[0,1]$$）划分为宽度为 $$\Delta$$ 的小区间，得到一个离散的随机变量 $$X_{\Delta}$$。
-   $$X_{\Delta}$$ 的可能取值为 $$\{0, \Delta, 2\Delta, \ldots, k\Delta, \ldots, 1\}$$ (其中 $$N=1/\Delta$$ 是区间个数)。
-   如果 $$X \sim U[0,1]$$，则 $$X_{\Delta}$$ 在这些离散点上近似均匀分布，每个点的概率 $$P(X_\Delta = k\Delta) \approx f(k\Delta)\Delta = 1 \cdot \Delta = \Delta$$ (对于 $$N = 1/\Delta$$ 个点)。
-   离散化后的熵 $$H(X_{\Delta})$$：
    $$H(X_{\Delta}) \approx \sum_{k=1}^{N} - \Delta \log_2 \Delta = - N \cdot \Delta \log_2 \Delta = - (1/\Delta) \cdot \Delta \log_2 \Delta = -\log_2 \Delta = \log_2(1/\Delta)$$。
-   当离散化的精度提高，即 $$\Delta \to 0$$ 时，$$N \to \infty$$，于是 $$H(X_{\Delta}) \to \infty$$。
-   **结论**：直接将离散熵的概念应用于精度无限提高的离散化连续随机变量，会导致熵趋于无穷大。这意味着精确编码一个连续随机变量理论上需要无穷多比特。

#### 2. 连续随机变量编码的物理意义
-   教师强调：对于连续随机变量，不存在“最短平均码长”这样的概念（以精确表示为目标时），因为要精确表示一个连续值，码长必然是无穷的。

### 2. 微分熵的定义与推导

#### 1. 形式上的推广 (动机)：
-   尽管物理意义上精确编码连续变量的码长是无穷的，我们能否从**形式上**将离散熵的定义推广到连续情况，以得到一个有用的量度？

#### 2. 微分熵的定义 ($$h(X)$$)：
-   **类比**：离散熵 $$H(X) = -\sum P(x) \log_2 P(x)$$。对于连续随机变量 $$X$$，其特征由**概率密度函数 (Probability Density Function, PDF)** $$f(x)$$ 描述。求和 $$\sum$$ 对应积分 $$\int$$，概率质量 $$P(x)$$ 对应 $$f(x)dx$$ 中的 $$f(x)$$ 部分（$$dx$$ 被吸收到积分中）。
-   微分熵 (Differential Entropy)，通常用小写 $$h(X)$$ 表示：
$$
h(X) = \int_{-\infty}^{\infty} f(x) \log_2 \frac{1}{f(x)} dx = - \int_{-\infty}^{\infty} f(x) \log_2 f(x) dx
$$
    （积分上下限默认为随机变量的支撑集）
-   **重要说明**：$$h(X)$$ **不是**表示编码连续随机变量 $$X$$ 所需的平均比特数。它只是形式上与 $$H(X)$$ 相似。

#### 3. 微分熵与离散熵的关系
-   对于一个连续随机变量 $$X$$ (PDF $$f(x)$$)，将其支撑集划分为宽度为 $$\Delta$$ 的小区间，中心点为 $$x_i = i\Delta$$。
-   离散化随机变量 $$X_{\Delta}$$ 取值 $$x_i$$ 的概率 $$P(X_{\Delta}=x_i) \approx f(x_i)\Delta$$。
-   $$H(X_{\Delta}) = -\sum_i P(X_{\Delta}=x_i) \log_2 P(X_{\Delta}=x_i)$$
-   $$\approx -\sum_i (f(x_i)\Delta) \log_2 (f(x_i)\Delta)$$
-   $$= -\sum_i f(x_i)\Delta [\log_2 f(x_i) + \log_2 \Delta]$$
-   $$= -\sum_i (f(x_i)\log_2 f(x_i))\Delta - \sum_i (f(x_i)\Delta) \log_2 \Delta$$
-   当 $$\Delta \to 0$$ 时，$$\sum \dots \Delta \to \int \dots dx$$:
    -   第一项 $$\to - \int f(x)\log_2 f(x) dx = h(X)$$
    -   第二项中 $$\sum_i f(x_i)\Delta \to \int f(x)dx = 1$$。所以第二项为 $$- (1) \cdot \log_2 \Delta = -\log_2 \Delta$$。
-   因此，得到近似关系 (对于足够小的 $$\Delta$$)：
$$
H(X_{\Delta}) \approx h(X) - \log_2 \Delta = h(X) + \log_2 \frac{1}{\Delta}
$$
-   这个关系揭示了 $$H(X_{\Delta})$$ 趋于无穷大的原因：因为当 $$\Delta \to 0$$ 时，$$\log_2 \frac{1}{\Delta} \to \infty$$。
-   $$h(X)$$ 可以看作是 $$H(X_{\Delta})$$ 中除去 $$\log_2 \frac{1}{\Delta}$$ 这一项后剩下的、与 $$\Delta$$ 无关的部分。它描述了随机变量 $$X$$ 的概率分布 $$f(x)$$ 的“形状”所带来的相对不确定性。

### 3. 微分熵的性质

#### 1. 与离散熵的对比 (平移与缩放)
-   **离散随机变量 (discrete r.v.) X**
    -   概率分布 $$P = (p_1, p_2, \ldots, p_{| \mathcal{X} |})$$
    -   平移 (Translation): 若 $$Y = X+c$$ (c 是常数)，则 $$H(Y) = H(X)$$ (熵不变，概率分布不变)。
    -   缩放 (Scaling): 若 $$Z = aX$$ ($$a \ne 0$$, $$a$$ 是常数)，则 $$H(Z) = H(X)$$ (熵不变，概率分布不变)。
-   **连续随机变量 (continuous r.v.) X** (PDF $$f_X(x)$$)
    -   **平移 (Translation)**: 若 $$Y = X+c$$
        PDF of Y: $$f_Y(y) = f_X(y-c)$$
$$
h(Y) = -\int f_Y(y) \log_2 f_Y(y) dy = -\int f_X(y-c) \log_2 f_X(y-c) dy
$$
        令 $$x = y-c$$, $$dx = dy$$
$$
h(Y) = -\int f_X(x) \log_2 f_X(x) dx = h(X)
$$
        所以，**微分熵具有平移不变性：$$h(X+c) = h(X)$$**。

    -   **缩放 (Scaling)**: 若 $$Y = aX$$ ($$a \ne 0$$)
        PDF of Y: $$f_Y(y) = \frac{1}{|a|} f_X(\frac{y}{a})$$
$$
h(Y) = -\int f_Y(y) \log_2 f_Y(y) dy = -\int \frac{1}{|a|}f_X(\frac{y}{a}) \log_2 \left[\frac{1}{|a|}f_X(\frac{y}{a})\right] dy
$$
        令 $$x = y/a$$, $$y = ax$$, $$dy = |a| dx$$ (若 $$a>0, dy=adx$$; 若 $$a<0, dy=-adx=|a|dx$$ if we integrate from $$-\infty$$ to $$\infty$$)
        Assume $$a>0$$ for simplicity in $$\log$$:
$$
h(Y) = -\int \frac{1}{a}f_X(x) \left[\log_2(\frac{1}{a}) + \log_2 f_X(x)\right] (a dx)
$$
$$
h(Y) = -\int f_X(x) \log_2 f_X(x) dx - \int f_X(x) \log_2(\frac{1}{a}) dx
$$
$$
h(Y) = h(X) - \log_2(\frac{1}{a}) \int f_X(x) dx
$$
        $$h(Y) = h(X) + \log_2 a \quad$$ (因为 $$\int f_X(x)dx = 1$$)
        对于任意 $$a \ne 0$$，结果是 $$h(aX) = h(X) + \log_2 |a|$$。
        所以，**微分熵在缩放变换下会改变：$$h(aX) = h(X) + \log_2 |a|$$**。
        这与离散熵不同。$$\log_2 |a|$$ 项也呼应了 $$H(X_\Delta) \approx h(X) - \log_2 \Delta$$ 中的 $$\log_2 \Delta$$ 项 (可看作 $$a=1/\Delta$$ 的 scaling，其中 $$\Delta$$ 代表量化单位的“尺寸”)。

#### 2. 微分熵可以为负 (Can Be Negative)
-   与离散熵恒为非负 ($$H(X) \ge 0$$) 不同，微分熵 $$h(X)$$ **可以取负值**。
-   **例1：均匀分布 $$X \sim U[0,a]$$**
    -   其PDF为 $$f(x) = 1/a$$ for $$x \in [0,a]$$，$$0$$ otherwise.
    -   $$h(X) = -\int_0^a \frac{1}{a} \log_2 \frac{1}{a} \, dx = - (\frac{1}{a} \log_2 \frac{1}{a}) \int_0^a dx = - (\frac{1}{a} \log_2 \frac{1}{a}) a = -\log_2 \frac{1}{a} = \log_2 a$$
    -   如果 $$a=1$$ (即 $$X \sim U[0,1]$$)，则 $$h(X) = \log_2 1 = 0$$。
    -   如果 $$a < 1$$ (例如 $$X \sim U[0, 0.5]$$)，则 $$h(X) = \log_2 0.5 = -1 < 0$$。
-   **例2：高斯分布 (Gaussian/Normal distribution) $$X \sim \mathcal{N}(\mu, \sigma^2)$$**
    -   其微分熵为 $$h(X) = \frac{1}{2} \log_2 (2\pi e \sigma^2)$$。
    -   如果方差 $$\sigma^2$$ 足够小，使得 $$2\pi e \sigma^2 < 1$$，那么 $$\log_2 (2\pi e \sigma^2)$$ 就会是负数，从而 $$h(X) < 0$$。
        ($$2\pi e \approx 2 \times 3.14159 \times 2.71828 \approx 17.08$$)。所以当 $$\sigma^2 < 1/(2\pi e) \approx 0.0585$$ 时，$$h(X) < 0$$。
-   这进一步说明微分熵的物理意义与离散熵（作为绝对信息量或不确定性的度量）有所不同。它更像是一个相对的量。

### 4. 总结与警示
-   微分熵不是一个“完美的”定义，它的物理含义不像离散熵那样清晰直接（不直接表示编码特定精度所需的总比特数，而是与量化精度有关的一个分量）。
-   在使用微分熵时需要小心，尤其是在进行变换（如缩放）时，其性质与离散熵不同。
-   尽管如此，微分熵仍然是一个非常有用的数学工具，在后续的信道容量（特别是高斯信道）等概念中会用到。许多涉及熵的差值或比率的结论（如互信息、条件微分熵）对于连续变量依然有意义。

---

## V. 布置的习题与思考

### 1. 上次课遗留的思考题：

-   KL散度 $$D_{KL}(P||Q)$$ 与 L1范数 $$||P-Q||_1$$ 之间的关系。（见第二部分解答）
-   KL散度的凸性：$$D_{KL}(P||Q)$$ 关于 $$(P,Q)$$ 对是联合凸函数。 (作为已知性质提及)

### 2. 本节课隐含或建议的思考/推导：

-   **高斯分布的微分熵推导**：
    对于 $$X \sim \mathcal{N}(\mu, \sigma^2)$$，其PDF为 $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$。
    推导其微分熵 $$h(X) = \frac{1}{2} \log_2 (2\pi e \sigma^2)$$。
    - *解答思路*:
$$
h(X) = - \int_{-\infty}^{\infty} f(x) \log_2 f(x) dx
$$
$$
\log_2 f(x) = \log_2 \left( (2\pi\sigma^2)^{-1/2} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \right) = -\frac{1}{2}\log_2(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} \log_2 e
$$
$$
h(X) = - \int f(x) \left[ -\frac{1}{2}\log_2(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} \log_2 e \right] dx
$$
$$
= \frac{1}{2}\log_2(2\pi\sigma^2) \int f(x)dx + (\log_2 e) \int f(x) \frac{(x-\mu)^2}{2\sigma^2} dx
$$
        $$\int f(x)dx = 1$$.
        $$\int f(x) (x-\mu)^2 dx = E[(X-\mu)^2] = \sigma^2$$.
        $$h(X) = \frac{1}{2}\log_2(2\pi\sigma^2) + (\log_2 e) \frac{\sigma^2}{2\sigma^2} = \frac{1}{2}\log_2(2\pi\sigma^2) + \frac{1}{2}\log_2 e = \frac{1}{2}\log_2(2\pi e \sigma^2)$$.

-   **验证微分熵的缩放性质**：
    若 $$Y = aX$$ ($$a \ne 0$$)，验证 $$h(Y) = h(X) + \log_2 |a|$$。（已在第四部分推导）

---

## VI. 总结与展望

本节课首先回顾了联合熵、条件熵、互信息和KL散度的核心概念及其关系。接着，针对编码效率问题和随机过程的信息度量，引入了**熵率**的概念，阐明了其两种定义及在i.i.d.信源和一般平稳信源下的意义，解释了熵作为平均码长下确界的深层含义。

随后，课程从离散熵向连续随机变量过渡，指出了直接应用离散熵概念到连续情况时熵会趋于无穷大的问题。为了处理连续随机变量，引入了**微分熵**作为离散熵在形式上的推广。我们详细讨论了微分熵的定义、其与离散熵在量化近似下的关系 ($$H(X_\Delta) \approx h(X) - \log_2 \Delta$$)，以及微分熵的重要性质（平移不变性、缩放变换性、可为负值）。通过均匀分布和高斯分布的例子，具体展示了微分熵的计算和特性。

虽然微分熵本身不直接表示绝对信息量，但它在信息论中，特别是在分析连续信源和信道（如高斯信道容量）时，是一个不可或缺的工具。后续课程将继续运用这些概念。