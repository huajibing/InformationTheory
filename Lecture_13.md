# 第13节：信道编码定理（逆定理），Fano不等式的应用，Fisher信息与Cramer-Rao不等式

---

## 〇、引言

在先前的课程中（如第10、11、12讲），我们已经探讨了信道的基本模型、信道容量的定义与计算方法，学习了渐进均分割特性 (AEP) 及其在证明香农信道编码定理（可达性部分）中的核心作用。我们理解到，当信息传输速率低于信道容量时，理论上存在编码方案能够实现任意低的传输错误概率。

本讲我们将继续深入信道编码理论，重点完成信道编码定理的另一半——***逆定理 (Converse)*** 的证明。我们将借助强大的***Fano 不等式 (Fano Inequality)***，揭示当传输速率试图超越信道容量时，为何可靠通信不再可能。随后，课程将转换到一个新的但同样重要的领域：参数估计的理论极限，引入***Fisher 信息 (Fisher Information)*** 和***Cramer-Rao 不等式 (Cramer-Rao Inequality)***，它们为衡量数据中参数信息量以及无偏估计的精度下限提供了基本工具。

---

## I. 信道编码定理 (Converse Part & Review)

### 1.1 信道编码定理的完整陈述

香农信道编码定理是信息论的核心成果，它精确地刻画了在有噪信道上可靠传输信息的速率极限。

*   **基本设定：**
    *   考虑一个离散无记忆信道 (DMC)，其输入为随机变量 $$X \in \mathcal{X}$$，输出为随机变量 $$Y \in \mathcal{Y}$$，信道特性由转移概率 $$P(Y|X)$$ 描述。
    *   信道容量 $$C = \max_{P(X)} I(X;Y)$$，单位为比特/信道使用。
    *   一个 $$(M, n)$$ 码由一个包含 $$M=2^{nR}$$ 个码字 $$c_1, \dots, c_M$$ 的码本和相应的译码规则构成，其中 $$n$$ 为码长，$$R = \frac{\log_2 M}{n}$$ 为编码速率。

*   **定理陈述：**
    1.  ***可达性 (Achievability / Direct Part):*** 对于任意速率 $$R < C$$ 和任意 $$\delta > 0$$，当码长 $$n$$ 足够大时，存在一个 $$(2^{nR}, n)$$ 码，其最大错误概率 (或平均错误概率) $$P_e^{(n)} < \delta$$。这意味着可以通过选择足够长的码字，以低于信道容量的任意速率进行可靠通信。
        *   *证明概要 (回顾 Lecture 11 & 12):* 香农的证明采用了随机编码方法。从达到容量的输入分布 $$P(X)$$ 中独立随机生成 $$2^{nR}$$ 个码字构成码本。译码时，接收端寻找与接收序列 $$Y^n$$ 唯一构成联合典型序列的码字。利用 AEP 和 Joint AEP 的性质，可以证明当 $$R < C$$ 时，平均错误概率随 $$n$$ 的增加而指数级趋于0。

    2.  ***逆定理 (Converse Part):*** 对于任意速率 $$R > C$$，任何 $$(2^{nR}, n)$$ 码的平均错误概率 `$$\bar{P}_e^{(n)}$$` (以及最大错误概率 $$P_{e,max}^{(n)}$$) 都有一个正的下界，即存在某个 $$\epsilon_0 > 0$$，使得对于足够大的 $$n$$，`$$\bar{P}_e^{(n)} \ge \epsilon_0$$`。这意味着如果传输速率超过信道容量，错误概率不能任意小，无法实现可靠通信。

### 1.2 错误概率的若干定义

在讨论信道编码性能时，有几种常用的错误概率定义：

1.  ***条件错误概率 (Conditional Error Probability):*** $$P_{e,i}$$
    指当发送第 $$i$$ 个消息 $$m_i$$ (对应码字 $$c_i$$) 时，译码器译码错误的概率。
$$
P_{e,i} = P(\hat{M} \neq m_i | M=m_i)
$$
    其中 $$\hat{M}$$ 是译码得到的消息。

2.  ***平均错误概率 (Average Error Probability):*** `$$\bar{P}_e^{(n)}$$`
    假设所有 $$M=2^{nR}$$ 个消息是等可能发送的，平均错误概率是在所有消息上的错误概率的平均值。
$$
\bar{P}_e^{(n)} = \frac{1}{M} \sum_{i=1}^{M} P_{e,i}
$$
    这是信道编码定理中经常分析的错误概率。

3.  ***最大错误概率 (Maximum Error Probability / Worst-case Error Probability):*** $$P_{e,max}^{(n)}$$
    指在所有可能发送的消息中，条件错误概率最大的那个。
$$
P_{e,max}^{(n)} = \max_{1 \le i \le M} P_{e,i}
$$
    这是对编码性能最强的保证。如果最大错误概率趋于0，那么平均错误概率和所有条件错误概率也都趋于0。

*注：在随机编码的证明中，这些错误概率通常是指在随机选择码本这一操作下的期望错误概率。如果期望平均错误概率趋于0，则至少存在一个固定的码本使得其平均错误概率趋于0。*

### 1.3 Fano 不等式

*Fano 不等式 (Fano Inequality)* 是信息论中的一个重要结果，它建立了基于观测估计随机变量时，估计错误概率与相应条件熵之间的定量关系。

*   **定理陈述 (Fano's Inequality):**
    设 $$X$$ 为一随机变量，取值于有限集合 $$\mathcal{X}$$。$$\hat{X}=g(Y)$$ 是基于对另一随机变量 $$Y$$ 的观测而对 $$X$$ 做出的估计。令 $$P_e = P(X \neq \hat{X})$$ 为估计的错误概率。则有：
$$
H(X|Y) \le H_b(P_e) + P_e \log_2(|\mathcal{X}|-1)
$$
    其中 $$H_b(P_e) = -P_e \log_2 P_e - (1-P_e) \log_2 (1-P_e)$$ 是二元熵函数。
    一个更常用（也更宽松，但在证明逆定理时足够用）的形式是：
$$
P_e \ge \frac{H(X|Y) - 1}{\log_2|\mathcal{X}|}
$$
    或者写为：
$$
H(X|Y) \le 1 + P_e \log_2|\mathcal{X}|
$$
    *(老师使用的是 $$P_e \ge \frac{H(X|Y) - 1}{\log_2|\mathcal{X}|}$$ 形式，分子中的“-1”对应于 $$H_b(P_e)$$ 中的一项。)*

*   **意义：**
    Fano 不等式表明，如果估计错误概率 $$P_e$$ 很小，那么观测到 $$Y$$ 之后关于 $$X$$ 的剩余不确定性 $$H(X|Y)$$ 也必定很小。反之，如果 $$H(X|Y)$$ 很大，那么 $$P_e$$ 不可能很小。这为从熵的角度分析错误概率提供了桥梁。

### 1.4 信道编码定理（逆定理）的证明

我们将使用 Fano 不等式来证明：如果编码速率 $$R$$ 大于信道容量 $$C$$，则平均错误概率 `$$\bar{P}_e^{(n)}$$` 不能任意小。

*   **证明思路：**
    1.  建立消息 $$M$$ 和接收序列 $$Y^n$$ 之间的条件熵 $$H(M|Y^n)$$ 与 $$R$$ 和 $$C$$ 的关系。
    2.  应用 Fano 不等式，将 `$$\bar{P}_e^{(n)}$$` 与 $$H(M|Y^n)$$ 联系起来。
    3.  结合 $$R>C$$ 的条件，导出 `$$\bar{P}_e^{(n)}$$` 的一个正下界。

*   **关键步骤与推导：**

    1.  **消息熵 $$H(M)$$:**
        假设有 $$M = 2^{nR}$$ 个等可能的消息。则消息源的熵为：
$$
H(M) = \log_2(2^{nR}) = nR
$$
    2.  **互信息 $$I(M; Y^n)$$ 的上界:**
        消息 $$M$$ 经过编码器得到码字 $$X^n(M)$$（简记为 $$X^n$$），再通过信道 $$P(Y|X)$$ 得到接收序列 $$Y^n$$。这是一个马尔可夫链: $$M \rightarrow X^n(M) \rightarrow Y^n$$。
        根据*数据处理不等式 (Data Processing Inequality)*，有：
$$
I(M; Y^n) \le I(X^n(M); Y^n)
$$
        对于离散无记忆信道 (DMC)，根据信道容量的定义以及互信息的链式法则和独立性：
$$
I(X^n(M); Y^n) = H(Y^n) - H(Y^n|X^n(M))
$$
$$
H(Y^n|X^n(M)) = \sum_{i=1}^n H(Y_i | Y_1, \dots, Y_{i-1}, X^n(M)) = \sum_{i=1}^n H(Y_i | X_i(M))
$$
        (由于信道无记忆且 $$X^n(M)$$ 给定时 $$Y_i$$ 只依赖 $$X_i(M)$$)。
$$
I(X^n(M); Y^n) = H(Y^n) - \sum_{i=1}^n H(Y_i | X_i(M)) \le \sum_{i=1}^n H(Y_i) - \sum_{i=1}^n H(Y_i | X_i(M)) = \sum_{i=1}^n I(X_i(M); Y_i)
$$
        由于 $$I(X_i; Y_i) \le C$$ (单个信道使用的容量)，所以：
$$
I(X^n(M); Y^n) \le \sum_{i=1}^n C = nC
$$
        因此，我们得到：
$$
I(M; Y^n) \le nC
$$
    3.  **条件熵 $$H(M|Y^n)$$ 的下界:**
        根据条件熵的定义 $$H(M|Y^n) = H(M) - I(M; Y^n)$$，代入上述结果：
$$
H(M|Y^n) \ge nR - nC = n(R-C)
$$
    4.  **应用 Fano 不等式:**
        将 Fano 不等式 $$P_e \ge \frac{H(X|Y) - 1}{\log_2|\mathcal{X}|}$$ 应用于当前场景：
        *   $$X \rightarrow M$$ (发送的消息)
        *   $$Y \rightarrow Y^n$$ (接收的序列)
        *   $$\hat{X} \rightarrow \hat{M}$$ (译码得到的消息)
        *   $$P_e \rightarrow \bar{P}_e^{(n)}$$ (平均错误概率)
        *   $$|\mathcal{X}| \rightarrow |M| = 2^{nR}$$ (消息总数)
        于是，Fano 不等式变为：
$$
\bar{P}_e^{(n)} \ge \frac{H(M|Y^n) - 1}{\log_2(2^{nR})} = \frac{H(M|Y^n) - 1}{nR}
$$
    5.  **推导错误概率的下界:**
        将 $$H(M|Y^n) \ge n(R-C)$$ 代入 Fano 不等式：
$$
\bar{P}_e^{(n)} \ge \frac{n(R-C) - 1}{nR}
$$
$$
\bar{P}_e^{(n)} \ge \frac{n(R-C)}{nR} - \frac{1}{nR}
$$
$$
\bar{P}_e^{(n)} \ge \frac{R-C}{R} - \frac{1}{nR}
$$
        或者写为：
$$
\bar{P}_e^{(n)} \ge \left(1 - \frac{C}{R}\right) - \frac{1}{nR}
$$
    6.  **结论：**
        如果速率 $$R > C$$，那么 $$1 - \frac{C}{R} > 0$$ (因为 $$C \ge 0$$)。
        当码长 $$n \to \infty$$ 时，$$\frac{1}{nR} \to 0$$。
        因此，对于足够大的 $$n$$：
$$
\bar{P}_e^{(n)} \ge 1 - \frac{C}{R}
$$
        令 $$\epsilon_0 = 1 - \frac{C}{R}$$。由于 $$R > C \ge 0$$，则 $$\frac{C}{R} < 1$$，所以 $$\epsilon_0 > 0$$。
        这就证明了，当 $$R > C$$ 时，平均错误概率 `$$\bar{P}_e^{(n)}$$` 存在一个大于零的下界 $$\epsilon_0$$，不能任意趋近于0。因此，可靠通信是不可能的。

*   **直观解释：**
    当 $$R > C$$ 时，$$n(R-C) \to \infty$$ (as $$n \to \infty$$)。这意味着即使在接收到 $$Y^n$$ 之后，关于原始消息 $$M$$ 的不确定性 $$H(M|Y^n)$$ 仍然非常大（随 $$n$$ 线性增长）。如此大的不确定性使得译码器无法可靠地从 $$2^{nR}$$ 个可能的发送消息中唯一确定原始消息，因此必然导致一个不可忽略的错误概率。信息传输的请求量 ($$nR$$) 超过了信道的信息承载能力 ($$nC$$)。

### 1.5 (思考题解答) Fano不等式在逆定理证明中的核心作用

*   **问题:** 如何利用 Fano 不等式（即 $$P_e \ge \frac{H(X|Y) - 1}{\log|\mathcal{X}|}$$）来证明信道编码定理的第二部分（逆定理）？特别是，当 $$R > C$$ 时，为什么错误概率 $$Err$$（对应这里的 $$P_e$$）不能趋向于0，而是必须大于等于某个正数 $$\epsilon_0$$？

*   **解答：**
    上述 **1.4 节的证明过程** 本身就是对这个思考题的完整解答。Fano 不等式的核心作用体现在以下几个方面：
    1.  **定量联系：** 它将我们关心的错误概率 `$$\bar{P}_e^{(n)}$$` 与信息论中的核心量——条件熵 $$H(M|Y^n)$$ 定量地联系起来。
    2.  **下界提供：** 它为错误概率提供了一个基于条件熵的下界。
    3.  **桥梁作用：**
        *   一方面，通过信息论的基本不等式（数据处理不等式）和信道容量的定义，我们可以推导出当 $$R>C$$ 时 $$H(M|Y^n)$$ 的一个正的、随 $$n$$ 增长的下界 $$n(R-C)$$。这表明此时接收端对发送消息的不确定性很大。
        *   另一方面，Fano 不等式告诉我们，这种大的不确定性 $$H(M|Y^n)$$ 必然导致错误概率 `$$\bar{P}_e^{(n)}$$` 也有一个不可忽略的下界。
    具体来说，当 $$R>C$$ 时，$$H(M|Y^n) \ge n(R-C)$$。将其代入 Fano 不等式 `$$\bar{P}_e^{(n)} \ge \frac{H(M|Y^n)-1}{nR}$$`，得到 `$$\bar{P}_e^{(n)} \ge \frac{n(R-C)-1}{nR} = (1-\frac{C}{R}) - \frac{1}{nR}$$`。因为 $$1-\frac{C}{R} > 0$$ 且 $$\frac{1}{nR} \to 0$$ (as $$n \to \infty$$)，所以 `$$\bar{P}_e^{(n)}$$` 被一个正数 $$1-\frac{C}{R}$$ 从下方约束，无法趋向于0。

---

## II. Fisher 信息与 Cramer-Rao 不等式

在结束了信道编码定理的讨论后，我们转向统计推断领域的一个基本问题：如何衡量数据中包含的关于未知参数的信息量，以及参数估计的精度极限。

### 2.1 参数估计引论

*   **基本设定：**
    *   我们有一组观测样本 $$X = (X_1, X_2, \ldots, X_n)$$，这些样本通常假设是独立同分布 (i.i.d.) 的。
    *   这些样本来自于一个概率分布，其形式已知，但包含一个或多个未知参数 $$\theta$$。这个分布用概率密度函数 $$f(x; \theta)$$ (连续情况) 或概率质量函数 $$p(x; \theta)$$ (离散情况) 表示。参数 $$\theta$$ 可以是标量或向量。
*   **目标：**
    利用观测到的样本 $$X$$ 来估计 (estimate) 未知参数 $$\theta$$。
    估计量 (estimator) 是样本的函数，记为 `$$\hat{\theta}(X) = \hat{\phi}(X_1, \ldots, X_n)$$`。
*   ***无偏估计 (Unbiased Estimator):***
    一个估计量 `$$\hat{\theta}(X)$$` 如果其期望值等于参数 $$\theta$$ 的真值，则称其为无偏的：
$$
E_{\theta}[\hat{\theta}(X)] = \theta \quad \text{for all possible } \theta
$$
    其中期望 $$E_{\theta}[\cdot]$$ 是在给定参数真值为 $$\theta$$ 时，对所有可能的样本 $$X$$ 取的。
*   **估计量的优良性：**
    对于无偏估计量，我们希望其方差 `$$\text{Var}_{\theta}(\hat{\theta}(X))$$` 尽可能小。方差越小，估计量围绕真值的波动越小，估计越精确。Cramer-Rao 不等式正是为此提供了理论下限。

### 2.2 Fisher 信息 (Fisher Information)

*Fisher 信息 (Fisher Information)* 是衡量样本数据中关于未知参数 $$\theta$$ 的信息量的一个核心概念。

1.  ***Score Function (得分函数) $$S(X; \theta)$$:**
    得分函数定义为*对数似然函数 (log-likelihood function)* 关于参数 $$\theta$$ 的偏导数（假设 $$\theta$$ 为标量）：
$$
S(X; \theta) = \frac{\partial}{\partial \theta} \ln f(X; \theta)
$$
    其中 $$f(X; \theta)$$ 是样本 $$X$$ 的联合概率密度函数（如果是i.i.d.样本，$$f(X;\theta) = \prod_{i=1}^n f(x_i;\theta)$$, 则 $$\ln f(X;\theta) = \sum_{i=1}^n \ln f(x_i;\theta)$$）。

    *   **得分函数的期望 $$E_{\theta}[S(X; \theta)]$$**:
        在某些***正则条件 (regularity conditions)*** 下（主要是允许积分和微分运算交换顺序，且似然函数对 $$\theta$$ 可导），得分函数的期望为0。
        **证明 (以单个样本 $$x$$ 为例):**
$$
E_{\theta}[S(x; \theta)] = \int \left(\frac{\partial}{\partial \theta} \ln f(x; \theta)\right) f(x; \theta) dx
$$
$$
= \int \frac{1}{f(x; \theta)} \left(\frac{\partial f(x; \theta)}{\partial \theta}\right) f(x; \theta) dx = \int \frac{\partial f(x; \theta)}{\partial \theta} dx
$$
        在正则条件下，可交换积分和微分顺序：
$$
= \frac{\partial}{\partial \theta} \int f(x; \theta) dx
$$
        由于 $$\int f(x; \theta) dx = 1$$ (概率密度函数的归一性)，
$$
E_{\theta}[S(x; \theta)] = \frac{\partial}{\partial \theta} (1) = 0
$$
        对于 i.i.d. 样本 $$X=(X_1, \dots, X_n)$$, $$S(X;\theta) = \sum_i S(X_i;\theta)$$, 故 $$E_{\theta}[S(X;\theta)] = \sum_i E_{\theta}[S(X_i;\theta)] = 0$$。

2.  **Fisher 信息 $$I(\theta)$$:**
    Fisher 信息定义为得分函数 $$S(X; \theta)$$ 的方差：
$$
I(\theta) = \text{Var}_{\theta}(S(X; \theta))
$$
    由于 $$E_{\theta}[S(X; \theta)] = 0$$，方差也等于其二阶矩：
$$
I(\theta) = E_{\theta}[S(X; \theta)^2] = E_{\theta}\left[\left(\frac{\partial}{\partial \theta} \ln f(X; \theta)\right)^2\right]
$$
    *   **Fisher 信息的另一种等价形式:**
        在正则条件下，Fisher 信息还可以表示为对数似然函数关于参数 $$\theta$$ 的二阶偏导数的期望的负值：
$$
I(\theta) = -E_{\theta}\left[\frac{\partial^2}{\partial\theta^2} \ln f(X; \theta)\right]
$$
        *(此等价性的证明见 2.5 节课后习题提示)*

    *   **Fisher 信息对于 i.i.d. 样本的性质:**
        如果样本 $$X = (X_1, X_2, \ldots, X_n)$$ 是 $$n$$ 个独立同分布的观测，每个样本的 Fisher 信息为 $$I_1(\theta)$$ (即基于单个 $$X_i$$ 的 Fisher 信息)，则整个样本 $$X$$ 的 Fisher 信息 $$I_n(\theta)$$ 是单个样本 Fisher 信息的 $$n$$ 倍：
$$
I_n(\theta) = n I_1(\theta)
$$
        **证明:**
        $$S(X;\theta) = \sum_{i=1}^n S(X_i;\theta)$$. 由于 $$X_i$$ 独立且 $$E[S(X_i;\theta)]=0$$,
        $$I_n(\theta) = \text{Var}\left(\sum_{i=1}^n S(X_i;\theta)\right) = \sum_{i=1}^n \text{Var}(S(X_i;\theta)) = \sum_{i=1}^n I_1(\theta) = n I_1(\theta)$$.

    *   **Fisher 信息的意义：**
        Fisher 信息 $$I(\theta)$$ 度量了观测样本 $$X$$ 中所包含的关于未知参数 $$\theta$$ 的信息量。$$I(\theta)$$ 越大，意味着对数似然函数在真值 $$\theta$$ 附近越尖锐，从而参数 $$\theta$$ 越容易被精确估计。它反映了似然函数对参数变化的敏感程度。

### 2.3 Cramer-Rao 不等式 (Cramer-Rao Inequality)

*Cramer-Rao 不等式 (Cramer-Rao Inequality)* 给出了任何无偏估计量方差的一个下限，这个下限与 Fisher 信息相关。

*   **定理陈述 (标量参数 $$\theta$$):**
    设 `$$\hat{\theta}(X)$$` 是参数 $$\theta$$ 的任一无偏估计量，即 $$E_{\theta}[\hat{\theta}(X)] = \theta$$。在正则条件下，其方差满足：
$$
\text{Var}_{\theta}(\hat{\theta}(X)) \ge \frac{1}{I(\theta)}
$$
    其中 $$I(\theta)$$ 是基于样本 $$X$$ 的 Fisher 信息。
    *   如果 `$$\hat{\theta}(X)$$` 是基于 $$n$$ 个i.i.d.样本的估计量，则 $$I(\theta) = nI_1(\theta)$$，不等式变为：
$$
\text{Var}_{\theta}(\hat{\theta}(X)) \ge \frac{1}{n I_1(\theta)}
$$
*   **意义 (*Cramer-Rao Lower Bound, CRLB*):**
    Cramer-Rao 不等式表明，无论采用多么巧妙的无偏估计方法，其估计量的方差都不可能小于 Fisher 信息的倒数。这个下限 $$1/I(\theta)$$ 被称为 *Cramer-Rao 下限 (CRLB)*。
    *   如果一个无偏估计量的方差达到了 CRLB，则称该估计量是***有效估计量 (efficient estimator)***。这样的估计量在均方误差意义下是最优的无偏估计量。
    *   CRLB 为评估和比较不同无偏估计量的性能提供了一个基准。

### 2.4 多维参数的推广 (简述)

当参数 $$\theta = (\theta_1, \dots, \theta_d)^T \in \mathbb{R}^d$$ 是一个 $$d$$ 维向量时：

*   **Score Function 向量:**
    得分函数变成一个 $$d$$ 维梯度向量：
$$
S(X;\theta) = \nabla_{\theta} \ln f(X;\theta) = \left( \frac{\partial \ln f}{\partial \theta_1}, \dots, \frac{\partial \ln f}{\partial \theta_d} \right)^T
$$
    其期望 $$E_{\theta}[S(X;\theta)] = \mathbf{0}$$ (零向量)。

*   ***Fisher Information Matrix (FIM) $$I(\theta)$$`:**
    Fisher 信息推广为一个 $$d \times d$$ 的矩阵，称为 *Fisher 信息矩阵 (FIM)*：
    其 $$(j,k)$$ 元素为 $$I(\theta)_{jk} = E_{\theta}[S_j(X;\theta) S_k(X;\theta)]$$。
$$
I(\theta) = E_{\theta}[S(X;\theta) S(X;\theta)^T]
$$
    等价地，其 $$(j,k)$$ 元素为 $$I(\theta)_{jk} = -E_{\theta}\left[\frac{\partial^2 \ln f(X;\theta)}{\partial \theta_j \partial \theta_k}\right]$$。
$$
I(\theta) = -E_{\theta}[\nabla_{\theta}^2 \ln f(X;\theta)]
$$
    其中 $$\nabla_{\theta}^2 \ln f(X;\theta)$$ 是对数似然函数的 Hessian 矩阵。FIM 是对称且半正定的。对于 i.i.d. 样本，$$I_n(\theta) = n I_1(\theta)$$ 仍然成立 (矩阵意义下)。

*   **Cramer-Rao 不等式 (矩阵形式):**
    设 `$$\hat{\theta}(X)$$` 是 $$\theta$$ 的无偏估计向量，$$E_{\theta}[\hat{\theta}(X)] = \theta$$。其协方差矩阵为 `$$\text{Cov}_{\theta}(\hat{\theta}(X)) = E_{\theta}[(\hat{\theta}(X)-\theta)(\hat{\theta}(X)-\theta)^T]$$`。
    Cramer-Rao 不等式变为：
$$
\text{Cov}_{\theta}(\hat{\theta}(X)) \succeq [I(\theta)]^{-1}
$$
    这里的 $$\succeq$$ 表示左边的矩阵减去右边的矩阵是半正定的 (positive semi-definite)。这意味着对于任意向量 $$u$$，有 $$u^T \text{Cov}_{\theta}(\hat{\theta}(X)) u \ge u^T [I(\theta)]^{-1} u$$。
    特别地，对于参数 $$\theta_j$$ 的估计 `$$\hat{\theta}_j(X)$$`，有 `$$\text{Var}_{\theta}(\hat{\theta}_j(X)) \ge ([I(\theta)]^{-1})_{jj}$$` (即 Fisher 信息矩阵逆的第 $$j$$ 个对角元素)。

### 2.5 (课后习题选解/提示)

1.  **证明 Fisher 信息的两种表达方式是等价的:**
    即证明 $$E_{\theta}[(\frac{\partial}{\partial \theta} \ln f(X;\theta))^2] = -E_{\theta}[\frac{\partial^2}{\partial\theta^2} \ln f(X;\theta)]$$。
    **提示：** 从 $$E_{\theta}[\frac{\partial}{\partial \theta} \ln f(X;\theta)] = 0$$ 出发。
$$
\int \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right) f(X;\theta) dx = 0
$$
    在正则条件下，对上式两边关于 $$\theta$$ 再求一次偏导数：
$$
\frac{\partial}{\partial \theta} \int \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right) f(X;\theta) dx = 0
$$
$$
\int \left[ \frac{\partial^2}{\partial\theta^2} \ln f(X;\theta) \cdot f(X;\theta) + \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right) \cdot \frac{\partial f(X;\theta)}{\partial \theta} \right] dx = 0
$$
    注意到 $$\frac{\partial f(X;\theta)}{\partial \theta} = \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right) f(X;\theta)$$。代入上式：
$$
\int \left[ \frac{\partial^2}{\partial\theta^2} \ln f(X;\theta) \cdot f(X;\theta) + \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right)^2 f(X;\theta) \right] dx = 0
$$
$$
\int \left(\frac{\partial^2}{\partial\theta^2} \ln f(X;\theta)\right) f(X;\theta) dx + \int \left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right)^2 f(X;\theta) dx = 0
$$
$$
E_{\theta}\left[\frac{\partial^2}{\partial\theta^2} \ln f(X;\theta)\right] + E_{\theta}\left[\left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right)^2\right] = 0
$$
    移项即得：
$$
E_{\theta}\left[\left(\frac{\partial}{\partial \theta} \ln f(X;\theta)\right)^2\right] = -E_{\theta}\left[\frac{\partial^2}{\partial\theta^2} \ln f(X;\theta)\right]
$$
    即 $$I(\theta) = E_{\theta}[S(X;\theta)^2] = -E_{\theta}\left[\frac{\partial^2}{\partial\theta^2} \ln f(X;\theta)\right]$$。

2.  **理解 Cramer-Rao 不等式对于多维参数的推广形式：**
    `$$\text{Cov}(\hat{\theta}(X)) \succeq [I(\theta)]^{-1}$$` 意味着 Fisher 信息矩阵 $$I(\theta)$$ 的“大小”（在某种逆的意义下）限制了无偏估计协方差矩阵的“大小”。如果 $$I(\theta)$$ 很大（信息量多），那么 $$[I(\theta)]^{-1}$$ 就“小”，从而协方差矩阵的下限也“小”，意味着可能存在更精确的估计。特别是，每个参数分量 $$\theta_j$$ 的估计方差 `$$\text{Var}(\hat{\theta}_j)$$` 受限于 `$$([I(\theta)]^{-1})_{jj}$$`，这通常大于仅考虑 $$\theta_j$$ 的标量 Fisher 信息 $$I(\theta_j)$$ 的倒数 (除非其他参数已知或与 $$\theta_j$$ 的估计不相关)。这反映了同时估计多个参数时的额外不确定性。

---

## III. 总结

本讲内容跨越了信道编码理论的边界和统计推断的基础。

1.  ***信道编码定理（逆定理）：*** 我们通过 **Fano 不等式**严格证明了，当信息传输速率 $$R$$ 试图超过信道容量 $$C$$ 时，平均错误概率 `$$\bar{P}_e^{(n)}$$` 必然被一个正的常数所限制，无法实现任意可靠的通信。这与可达性部分共同构成了香农第二定理的完整图景，指明了可靠通信的理论极限。

2.  ***Fisher 信息与 Cramer-Rao 不等式：***
    *   ***Fisher 信息*** $$I(\theta)$$ 作为衡量数据中关于未知参数信息量的关键指标被引入，它与对数似然函数的曲率相关。
    *   ***Cramer-Rao 不等式*** `$$\text{Var}(\hat{\theta}) \ge 1/I(\theta)$$` 为任何无偏估计量的方差设定了一个不可逾越的下限 (CRLB)。
    这些概念是参数估计理论的基石，为评估统计推断方法的效率和探索最优估计提供了理论依据。

通过本讲的学习，我们不仅深化了对信息传输极限的理解，也初步接触了信息论思想在统计推断问题中的应用。

---