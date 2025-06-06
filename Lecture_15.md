# 第15节：随机化通信协议，数学研究拾遗与最大熵原理

---

## 〇、课程引言

本节课程将首先回顾随机化通信协议在解决相等函数问题上的错误概率分析，展示其如何以较低的通信代价实现高概率的正确性。随后，我们将简要探讨素数定理、哥德巴赫猜想等数学问题以及当前数学研究的一些现状与挑战，并提及人工智能在数学研究中的潜在应用。课程的核心内容将转向介绍和证明***最大熵原理 (Maximum Entropy Principle)***，阐述在给定部分统计信息（如均值、协方差）的条件下，如何推断最符合这些信息的概率分布，并证明多维高斯分布是在给定均值和协方差矩阵时具有最大微分熵的分布。最后，我们将讨论最大熵原理在离散随机变量上的应用。

---

## I. 随机化通信协议回顾与分析 (Equality Function)

在之前的课程中（参考第14讲内容），我们引入了通信复杂度的概念，并讨论了确定性协议。本节将回顾一个针对***相等函数 (Equality Function)*** 的随机化通信协议，并重点分析其错误概率。

### 1.1 相等函数与协议回顾

*   **相等函数 $$f_{EQ}(x,y)$$**:
    Alice 持有 $$x \in \{0,1\}^n$$，Bob 持有 $$y \in \{0,1\}^n$$。
$$
f_{EQ}(x,y) = \begin{cases} 1 & \text{if } x=y \\ 0 & \text{if } x \neq y \end{cases}
$$
    确定性协议的通信复杂度为 $$n$$ 比特。

*   **随机化协议流程**:
    0.  **(预备阶段 - Alice 和 Bob 共同约定)**:
        选择一个足够大的素数 $$p$$ (例如 $$p \in [n^{100}, n^{101}]$$)。
    1.  **(Alice 的操作)**:
        将输入 $$x = (x_0, x_1, \dots, x_{n-1})$$ 视为多项式 $$f(Z) = \sum_{i=0}^{n-1} x_i Z^i$$ 的系数。
        Alice 随机选择一个整数 $$t \in \{0, 1, \dots, p-1\}$$ (即 $$t \in \mathbb{Z}_p$$)。
        Alice 计算 $$f(t) \pmod p$$。
    2.  **(Alice 的通信)**:
        Alice 将 $$t$$ 和 $$f(t) \pmod p$$ 发送给 Bob。
    3.  **(Bob 的操作)**:
        Bob 接收 $$t$$ 和 $$f(t)$$。
        将输入 $$y = (y_0, y_1, \dots, y_{n-1})$$ 视为多项式 $$g(Z) = \sum_{i=0}^{n-1} y_i Z^i$$ 的系数。
        Bob 计算 $$g(t) \pmod p$$。
    4.  **(Bob 的输出)**:
        如果 $$f(t) \equiv g(t) \pmod p$$，则输出 $$x=y$$。
        如果 $$f(t) \not\equiv g(t) \pmod p$$，则输出 $$x \neq y$$。

*   **通信代价**:
    Alice 发送 $$t$$ ($$O(\log p)$$ 比特) 和 $$f(t)$$ ($$O(\log p)$$ 比特)。
    总通信代价为 $$O(\log p)$$。若 $$p \approx n^{100}$$，则通信代价为 $$O(\log (n^{100})) = O(100 \log n) = O(\log n)$$ 比特。

### 1.2 错误概率分析

*   > **(教师板书 - 回顾)**
    *   **错误 (Error) 的情况**: 当 Alice 的输入 $$x \neq y$$ Bob 的输入，但协议错误地判断 $$x=y$$。
        *   这发生在 $$f(t) \equiv g(t) \pmod p$$ 的情况下。
        *   令 $$h(Z) = f(Z) - g(Z)$$。由于 $$x \neq y$$，所以 $$h(Z)$$ 是一个***非零多项式 (non-zero polynomial)***，其***次数 (degree)*** $$\le n-1$$。
            *   因为 $$x, y \in \{0,1\}^n$$ 且 $$x \neq y$$，所以至少存在一个 $$i$$ 使得 $$x_i \neq y_i$$。这意味着 $$x_i - y_i \not\equiv 0 \pmod p$$ (当 $$p>1$$ 时，因为 $$x_i-y_i$$ 只能是 $$1$$ 或 $$-1$$)。因此 $$h(Z)$$ 的系数不全为零。
        *   错误发生当且仅当 $$h(t) \equiv 0 \pmod p$$，即 $$t$$ 是多项式 $$h(Z)$$ 在域 $$\mathbb{Z}_p$$ 上的一个根 (zero/root)。

*   **错误概率分析**:
    *   根据代数基本定理的一个推论：一个在域（如 $$\mathbb{Z}_p$$）上的 $$d$$ 次非零多项式至多有 $$d$$ 个根。
    *   因此，$$h(Z)$$ (次数 $$\le n-1$$) 在 $$\mathbb{Z}_p$$ 上至多有 $$n-1$$ 个根。
    *   $$t$$ 是从集合 $$\{0, 1, \dots, p-1\}$$ (包含 $$p$$ 个元素) 中均匀随机选取的。
    *   所以，当 $$x \neq y$$ 时，协议发生错误的概率为：
$$
P(\text{error}) = P(f(t) \equiv g(t) \pmod p \mid x \neq y) = P(h(t) \equiv 0 \pmod p \mid x \neq y)
$$
$$
P(\text{error}) \le \frac{\text{number of roots of } h(Z) \text{ in } \mathbb{Z}_p}{p} \le \frac{n-1}{p}
$$
    *   如果选择 $$p \approx n^{100}$$ (一个非常大的素数)，那么：
$$
P(\text{error}) \le \frac{n-1}{n^{100}} \approx \frac{n}{n^{100}} = \frac{1}{n^{99}}
$$
        这是一个非常小的概率，随着 $$n$$ 的增大而迅速趋向于0。
    *   因此，该随机化协议是***高概率正确 (high probability correct)*** 的。

*   **正确性分析 - 完整**:
    *   **Case 1: $$x = y$$**
        此时，$$f(Z)$$ 和 $$g(Z)$$ 是完全相同的多项式。因此，对于任何 $$t$$，都有 $$f(t) \equiv g(t) \pmod p$$。协议总是输出 $$x=y$$，是**正确的 (correct)**。
    *   **Case 2: $$x \neq y$$**
        此时，我们希望协议输出 $$x \neq y$$。错误发生在 $$f(t) \equiv g(t) \pmod p$$。如上分析，该错误概率 $$P(\text{error}) \le \frac{n-1}{p}$$，可以做得非常小。

*   **教师总结**: 这个随机化协议能够以很高的概率给出正确结果，并且通信代价仅为 $$O(\log n)$$，远小于确定性协议的 $$O(n)$$。

---

---

## II. 数学漫谈与研究现状

### 2.1 素数定理与哥德巴赫猜想

*   > **(教师板书 - 黑板左侧)**
    *   ***素数定理 (Prime Number Theorem - PNT)***:
    >   `#primes ≤ m` $$\approx \frac{m}{\ln m}$$
    >   (小于等于 $$m$$ 的素数个数 $$\pi(m)$$ 近似于 $$\frac{m}{\ln m}$$)。
    >   这个定理描述了素数的分布密度，保证了在需要时（如随机化协议中）可以找到足够大的素数。
    *   ***哥德巴赫猜想 (Goldbach Conjecture)***:
    >   $$\exists n_1, n_2$$ prime`
    >   $$m = n_1 + n_2$$
    >   $$m$$ is even`
    >   (任何一个大于2的偶数都可以表示为两个素数之和)。
*   教师提到陈景润在哥德巴赫猜想上的贡献（证明了“1+2”，即任何一个充分大的偶数都可以表示为一个素数与一个至多是两个素数乘积之和）。

### 2.2 数学研究现状与论文评审

*   教师引申讨论了当前数学研究的一些特点：
    *   **证明的复杂性与长度**: 许多重要的数学成果，其证明过程非常复杂和冗长。
    *   ***同行评审 (Peer Review)*** 的挑战:
        *   评审人通常没有足够的时间和精力去逐行检查非常长的证明（例如数百页）的每一个细节。
        *   评审主要关注论文的***高层思想 (high-level idea)*** 和***路线图 (roadmap)***，并抽查一些关键的引理或步骤。
        *   教师提到，有时一个非常长的证明（例如关于哥德巴赫猜想的某个尝试性证明）可能发表在网上（如 arXiv），但要获得顶级期刊的接受和广泛认可则非常困难，部分原因就是评审的难度。
    *   **正确性的不确定性**:
        *   即使是论文的作者本人，在面对极其复杂的证明时，也可能无法100%保证每一个细节都没有问题。
        *   数学界有时会出现已发表的论文后来被发现存在错误（gap 或致命错误）的情况。这在数学发展中是一个普遍现象，并非个例。

### 2.3 AI for Math (人工智能在数学研究中的应用)

*   教师提到自己目前也在进行一些 "AI for Math" 相关的研究工作。
*   思考方向：是否可以利用机器学习或人工智能的方法来辅助数学研究，例如：
    *   **定理证明 (Theorem Proving)**
    *   **发现新的数学规律或猜想 (Discovering new mathematical patterns or conjectures)**

---

## III. 最大熵原理 (Maximum Entropy Principle)

## III. 最大熵原理 (Maximum Entropy Principle)

### 3.1 问题引入与一维情况回顾

*   > **(教师板书)**
    > *   `Max Entropy.`
*   **问题设定**:
    *   假设有一个 $$d$$ 维随机向量 (d-dim random vector) $$X = (X_1, \dots, X_d)^T$$。
    *   我们已知关于这个随机向量的一些***部分信息 (partial information)***，通常体现为某些期望值的约束。例如：
        1.  它的***均值 (mean)***: $$E[X] = \vec{\mu}$$ (通常为了简化，会假设 $$\vec{\mu} = \vec{0}$$，如果不是零向量，可以通过平移 $$X' = X - \vec{\mu}$$ 来处理)。
        2.  它的***协方差矩阵 (covariance matrix)***: $$Cov(X) = E[(X-E[X])(X-E[X])^T] = \Sigma$$。
            (教师补充：当均值为零向量 $$E[X]=\vec{0}$$ 时，$$Cov(X) = E[XX^T]$$)。

*   **核心问题**
    > **(教师板书)**:
    > *   `Max Entropy distribution ?`
    *   在所有满足已知约束条件（例如，给定均值和协方差矩阵）的概率分布中，哪一个分布的***熵 (entropy)*** 最大？

*   **回顾一维情况 (1-dim case, $$d=1$$)**:
    *   如果 $$X$$ 是一维随机变量，已知其均值 $$\mu$$ 和方差 $$\sigma^2$$，那么使得其微分熵 $$h(X)$$ 最大的分布是***高斯分布 (Gaussian distribution) / 正态分布 (Normal distribution)*** $$N(\mu, \sigma^2)$$。

### 3.2 多维高斯分布的最大熵特性

*   **定理**: 对于 $$d$$ 维随机向量 $$X$$，在给定均值 $$E[X]=\vec{\mu}$$ 和协方差矩阵 $$Cov(X)=\Sigma$$ (其中 $$\Sigma$$ 是正定矩阵) 的条件下，使得其***微分熵 (differential entropy)*** $$h(X)$$ 最大的分布是***多维高斯分布 (Multivariate Gaussian distribution)***，即 $$X \sim N(\vec{\mu}, \Sigma)$$。
*   > **(教师板书 - 以均值为0为例)**:
    > *   `X d-dim random vector`
    > *   `E[X] = $$\vec{0}$$
    > *   `Cov(X) = $$\Sigma$$ (旁边有 $$E[XX^T]$$)
    > *   `Max Entropy distribution is $$N(\vec{0}, \Sigma)$$` (教师写为 `Th` (Theorem) $$N(0, \Sigma)$$)

### 3.3 最大熵原理的证明 (多维高斯情况)

我们将证明：若随机向量 $$X$$ 的均值为 $$\vec{0}$$，协方差为 $$\Sigma$$，则其微分熵 $$h(X)$$ 小于等于具有相同均值和协方差的多维高斯分布 $$Y \sim N(\vec{0}, \Sigma)$$ 的微分熵 $$h(Y)$$。

*   > **(教师板书 - 定理陈述)**:
    > *   `Thm`
    > *   `Let X be a d-dim random vector.`
    > *   `EX = $$\vec{0}$$`, `Cov(X) = $$\Sigma$$`
    > *   `Let Y ~ N($$\vec{0}$$, $$\Sigma$$)`
    > *   `Then $$h(X) \le h(Y)$$`

*   **证明过程 (利用KL散度)**:
    1.  令 $$f_X(t)$$ 为随机向量 $$X$$ 的概率密度函数 (PDF)，$$f_Y(t)$$ 为随机向量 $$Y \sim N(\vec{0}, \Sigma)$$ 的 PDF。
    2.  考虑 $$X$$ 和 $$Y$$ 之间的***KL散度 (Kullback-Leibler Divergence)*** 或相对熵 $$D(f_X || f_Y)$$ (简记为 $$D(X||Y)$$):
$$
D(X||Y) = \int f_X(t) \log \frac{f_X(t)}{f_Y(t)} dt
$$
        根据信息不等式 (Gibbs' inequality的连续版本)，我们知道 $$D(X||Y) \ge 0$$，等号成立当且仅当 $$f_X(t) = f_Y(t)$$ 几乎处处成立。
    3.  展开 KL 散度：
$$
D(X||Y) = \int f_X(t) \log f_X(t) dt - \int f_X(t) \log f_Y(t) dt
$$
$$
D(X||Y) = -h(X) - \int f_X(t) \log f_Y(t) dt
$$
    4.  由于 $$D(X||Y) \ge 0$$，我们有：
$$
-h(X) - \int f_X(t) \log f_Y(t) dt \ge 0
$$
$$
\implies h(X) \le - \int f_X(t) \log f_Y(t) dt
$$
    5.  现在我们需要计算积分项 $$- \int f_X(t) \log f_Y(t) dt$$。
        多维高斯分布 $$Y \sim N(\vec{0}, \Sigma)$$ 的 PDF 为：
$$
f_Y(t) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} t^T \Sigma^{-1} t\right)
$$
        其对数 $$\log f_Y(t)$$ (使用自然对数 $$\ln$$ 或以2为底的对数 $$\log_2$$，这里为与熵定义一致，假设是 $$\ln$$，若用 $$\log_2$$ 则熵单位为比特)：
$$
\ln f_Y(t) = \ln\left(\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}\right) - \frac{1}{2} t^T \Sigma^{-1} t
$$
$$
\ln f_Y(t) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma| - \frac{1}{2} t^T \Sigma^{-1} t
$$
    6.  计算积分 $$\int f_X(t) \ln f_Y(t) dt$$:
$$
\int f_X(t) \ln f_Y(t) dt = \int f_X(t) \left[-\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma| - \frac{1}{2} t^T \Sigma^{-1} t\right] dt
$$
        由于 $$\int f_X(t) dt = 1$$ (PDF的性质)，常数项可以提出：
$$
= -\left(\frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\Sigma|\right) \int f_X(t) dt - \frac{1}{2} \int f_X(t) (t^T \Sigma^{-1} t) dt
$$
$$
= -\left(\frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\Sigma|\right) - \frac{1}{2} E_X[t^T \Sigma^{-1} t]
$$
    7.  计算 $$E_X[t^T \Sigma^{-1} t]$$:
        $$t^T \Sigma^{-1} t$$ 是一个标量，等于其迹 $$tr(t^T \Sigma^{-1} t)$$。
        利用迹的轮换不变性 $$tr(ABC) = tr(BCA) = tr(CAB)$$：
        $$tr(t^T \Sigma^{-1} t) = tr(\Sigma^{-1} t t^T)$$
        所以，$$E_X[t^T \Sigma^{-1} t] = E_X[tr(\Sigma^{-1} t t^T)]$$
        由于期望和迹运算可以交换顺序 (线性性)：
        $$E_X[t^T \Sigma^{-1} t] = tr(E_X[\Sigma^{-1} t t^T]) = tr(\Sigma^{-1} E_X[t t^T])$$
        因为 $$E[X] = \vec{0}$$，所以 $$Cov(X) = E[XX^T] = \Sigma$$。在此处，由于积分变量是 $$t$$ 代表 $$X$$ 的实现，所以 $$E_X[t t^T] = \Sigma$$。
        $$E_X[t^T \Sigma^{-1} t] = tr(\Sigma^{-1} \Sigma) = tr(I_d) = d$$
        (其中 $$I_d$$ 是 $$d \times d$$ 的单位矩阵)。
    8.  将 $$E_X[t^T \Sigma^{-1} t] = d$$ 代回第6步的积分结果：
$$
\int f_X(t) \ln f_Y(t) dt = -\left(\frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\Sigma|\right) - \frac{1}{2}d
$$
        所以，
$$
- \int f_X(t) \ln f_Y(t) dt = \frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\Sigma| + \frac{1}{2}d
$$
$$
= \frac{1}{2} \ln((2\pi)^d |\Sigma|) + \frac{d}{2}\ln(e) = \frac{1}{2} \ln((2\pi e)^d |\Sigma|)
$$
        这正是多维高斯分布 $$Y \sim N(\vec{0}, \Sigma)$$ 的微分熵 $$h(Y)$$ 的表达式 (如果使用 $$\log_2$$，则为 $$\frac{1}{2} \log_2((2\pi e)^d |\Sigma|)$$ 比特)。
    9.  **结论**:
        我们已经证明了 $$- \int f_X(t) \ln f_Y(t) dt = h(Y)$$。
        代回第4步的不等式 $$h(X) \le - \int f_X(t) \ln f_Y(t) dt$$，得到：
$$
h(X) \le h(Y)
$$
        等号成立当且仅当 $$D(X||Y) = 0$$，即 $$f_X(t) = f_Y(t)$$ 几乎处处成立。这意味着 $$X$$ 自身也服从 $$N(\vec{0}, \Sigma)$$ 分布。

### 3.4 最大熵原理的其他应用示例 (离散情况)

最大熵原理不仅适用于连续随机变量，也适用于离散随机变量。

*   > **(教师板书 - 黑板右上角)**
    > *   `rev X discrete` (回顾X是离散型随机变量)
    > *   `s.t. pmf $$p_i$$` (其概率质量函数为 $$p_i = P(X=x_i)$$)
    > *   `EX = $$\mu$$` (给定均值 $$E[X] = \sum_i x_i p_i = \mu$$)
    > *   `C. $$e^{-\lambda t}$$` (教师提示最大熵分布的形式，具体指 $$p_i = C e^{-\lambda x_i}$$, $$C$$为归一化常数)

*   **一般形式**:
    如果对一个随机变量 $$X$$ (离散或连续)，我们有一些期望值形式的约束：
    $$E[\phi_j(X)] = \alpha_j$$ for $$j=1, \dots, k$$
    以及概率归一化约束 $$\sum p(x) = 1$$ 或 $$\int p(x)dx = 1$$。
    那么，使得熵 $$H(X)$$ (或 $$h(X)$$) 最大的概率分布 $$p(x)$$ (PMF或PDF) 通常具有以下指数形式：
$$
p(x) = \frac{1}{Z(\vec{\lambda})} \exp\left(-\sum_{j=1}^k \lambda_j \phi_j(x)\right)
$$
    其中 $$\lambda_j$$ 是与约束 $$\alpha_j$$ 对应的***拉格朗日乘子 (Lagrange Multipliers)***，而 $$Z(\vec{\lambda}) = \sum_x \exp(-\sum_j \lambda_j \phi_j(x))$$ (或积分形式) 是归一化常数，也称为***配分函数 (Partition Function)***。
    这种方法通常使用变分法结合拉格朗日乘子法来求解。

*   **示例：离散随机变量，定义在非负整数上，给定均值**
    *   设 $$X$$ 是一个离散随机变量，其取值范围 (Support) 为 $$\{0, 1, 2, \dots\}$$。
    *   约束条件：
        1.  $$\sum_{i=0}^\infty p_i = 1$$ (概率归一化)
        2.  $$E[X] = \sum_{i=0}^\infty i p_i = \mu$$ (给定均值 $$\mu > 0$$)
    *   根据上述一般形式，$$\phi_1(i) = i$$。最大熵分布的 PMF $$p_i$$ 具有形式 $$p_i \propto \exp(-\lambda i) = (e^{-\lambda})^i = q^i$$ (令 $$q=e^{-\lambda}$$)。
    *   这正是***几何分布 (Geometric distribution)*** 的形式: $$p_i = (1-q)q^i$$ for $$i=0,1,2,\dots$$。
        其中参数 $$q$$ 通过均值约束 $$\mu = \frac{q}{1-q}$$ 来确定。

*   **教师总结**: 最大熵原理是一个非常重要的思想和工具，它提供了一种在信息不足的情况下进行概率推断的原则性方法。它不仅在信息论中有用，在统计物理 (如玻尔兹曼分布)、机器学习 (如逻辑回归的最大熵模型 MaxEnt)、经济学等其他领域也有广泛应用。它主张，在满足所有已知约束的前提下，我们应该选择不作任何其他假设的分布，即最“不确定”或最“均匀”的分布，也就是熵最大的那个分布。

---

## IV. 总结与展望

本节课程首先深入分析了用于相等函数判断的随机化通信协议的错误概率，确认了其高效性。随后，通过对素数定理、哥德巴赫猜想和数学研究现状的讨论，拓宽了视野。核心部分详细介绍并证明了**最大熵原理**的一个重要实例：在给定均值和协方差的条件下，**多维高斯分布**具有最大的微分熵。我们还探讨了最大熵原理的一般形式及其在离散随机变量上的应用。

最大熵原理是构建统计模型和进行推断的一个基本准则，它鼓励我们选择在满足已知事实的前提下最为“无偏”的概率分布。

*本节课未明确布置新的习题。教师预祝同学们考试顺利。*

---