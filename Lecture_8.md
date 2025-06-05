**《信息论》第8讲 课堂笔记：信道编码、编码界与汉明码初步**

**〇、引言**

上节课（可能为第7讲）我们可能讨论了柯尔莫哥洛夫复杂度或最大熵原理等内容。本节课我们将进入**信道编码 (Channel Coding)** 的领域。信源编码的目标是去除冗余以实现高效表示，而信道编码则恰恰相反，它通过有策略地引入冗余来对抗信道噪声，以实现可靠通信。我们将探讨信道编码的动机、基本思想、核心的性能权衡（可靠性 vs. 效率），并介绍两个重要的理论界限：**球堆积界 (Sphere Packing Bound)** 作为码长下界，以及通过**概率方法 (Probabilistic Method)** 得到的**吉尔伯特-瓦尔沙莫夫界 (Gilbert-Varshamov Bound)** 作为码存在性的上界。最后，我们将初步引入一类重要的结构化码——**汉明码 (Hamming Code)**。

---

**一、信道编码 (Channel Coding) 导论 (源自 Part 1)**

1.  **上节回顾与本节引入 (源自 Part 1: 05:30 - 05:50)**
    *   **上节课可能内容回顾**: 可能涉及信源编码 (Source Coding)，核心是熵 (Entropy) 及其应用，旨在实现最小描述 (minimum description)。
    *   **本节课主题**: **信道编码 (Channel Coding)**。

2.  **信道编码的动机 (Motivation) (源自 Part 1: 05:50 - 07:20)**
    *   **动机1: 有噪信道 (Noisy Channel)**
        *   物理信道普遍存在噪声，导致接收信息与发送信息可能不一致。
        *   噪声是广义的，指任何导致传输错误的因素。
        *   **(板书)**: `Motivation: 1) Noisy Channel`
    *   **动机2: 通过算法减少/纠正错误 (Algorithm reduces error / corrects)**
        *   信道编码的目标是设计编码 (Encode) 和解码 (Decode) 算法，以对抗噪声，减少或纠正错误。
        *   **(板书)**: `2) Alg. reduces error (induced by noise) (corrects?)`
        *   **(板书)**: `Encode/Decode`

3.  **基本思想与简单示例：重复码 (Repetition Code) (源自 Part 1: 07:20 - 08:20)**
    *   **模型：二进制对称信道 (Binary Symmetric Channel - BSC)**
        *   输入符号: $\{0, 1\}$；输出符号: $\{0, 1\}$
        *   $P(0|0) = P(1|1) = 1-\epsilon$ (正确传输概率)
        *   $P(1|0) = P(0|1) = \epsilon$ (交叉错误概率, crossover probability)
        *   **(板书图示)**:
            ```
            0 --(1-ε)--> 0
            |             ^
            (ε)           (ε)
            v             |
            1 --(1-ε)--> 1
            (noisy channel)
            ```
        *   例如 $\epsilon = 0.1$。
    *   **重复码 (Repetition Code)**
        *   **(板书)**: `Repetition Code`
        *   **编码 (Encoding)**: 将1比特信息重复多次。
            *   $0 \rightarrow 000$
            *   $1 \rightarrow 111$
            **(板书)**: `Encoding: 0 -> 000, 1 -> 111`
        *   **解码 (Decoding)**: **多数表决 (Majority vote)** 或 **最近邻解码 (Nearest Neighbor Decoding)** (在BSC下两者等价)。
            *   例如，收到 `001`，解码为 `0`。
            **(板书)**: `Decoding: Majority vote, Nearest Neighbor`
        *   **效果**: 可以使得最终错误概率 $\epsilon'$ 小于原始信道错误概率 $\epsilon$。
            *   例如，对于3次重复码，解码错误发生在至少2个比特出错时。
                $\epsilon' = P(\text{2 errors}) + P(\text{3 errors}) = \binom{3}{2}\epsilon^2(1-\epsilon) + \binom{3}{3}\epsilon^3 = 3\epsilon^2(1-\epsilon) + \epsilon^3$。
                若 $\epsilon = 0.1$, $\epsilon' = 3(0.01)(0.9) + 0.001 = 0.027 + 0.001 = 0.028 < 0.1$。
            *   **(板书)**: `ε = 0.1 -> ε' < ε`
        *   理论上，当重复次数 $n \rightarrow \infty$，$\epsilon' \rightarrow 0$ (如果 $\epsilon < 1/2$)。

4.  **信道编码的核心权衡：可靠性 vs. 效率 (源自 Part 1: 08:20 - 结束)**
    *   **效率 (Efficiency)**
        *   重复码虽然提高了可靠性，但牺牲了效率。
        *   **码率 (Code Rate)**: $R = \frac{m}{n}$，其中 $m$ 是原始信息比特数，$n$ 是编码后码字的比特数。
        *   对于上述3次重复码，$m=1, n=3$, $R = 1/3$。
        *   **(板书)**: `Efficiency`
    *   **问题设定**:
        *   **消息空间 (Message space)**: $M = 2^m$ 个不同消息，每个消息用 $m$ 比特表示。
            **(板书)**: `Message space {0,1}^m ; M_1, M_2, ..., M_{2^m}`
        *   **码字空间 (Codeword space)**: 将 $2^m$ 个消息一对一映射到 $n$ 比特的码字 ($n > m$)。
            **(板书)**: `Codeword {0,1}^n ; C_1, C_2, ..., C_{2^m}`
        *   **纠错能力与汉明距离 (Hamming Distance)**:
            *   汉明距离 $d_H(C_i, C_j)$ 是指两个等长码字 $C_i$ 和 $C_j$ 之间对应位置上符号不同的位数。
            *   为了能够纠正 $t$ 个比特的错误，任意两个不同码字 $C_i$ 和 $C_j$ 之间的汉明距离必须满足：
                $$d_H(C_i, C_j) \ge 2t+1$$
                **(板书)**: `d_H(C_i, C_j) >= 2t+1 (bits) (correct t bit error)`
            *   **解释**: 以每个码字 $C_k$ 为中心，半径为 $t$ 的汉明球（包含所有与 $C_k$ 距离 $\le t$ 的 $n$ 比特序列）必须互不相交。这样，当接收到的序列落在某个球内时，可以唯一地解码为该球的球心码字，从而纠正了 $\le t$ 个错误。

---

**二、编码的理论界限**

1.  **下界 (Lower Bound) - 球堆积界 (Sphere Packing Bound / Hamming Bound) (源自 Part 1, Part 2 回顾)**
    *   **目标**: 给定纠错能力 $t$ 和消息数量 $2^m$，找到码长 $n$ 的一个下限。
        **(板书 Part 1)**: `Goal: Given t, given m (=> 2^m messages), finds a lower bound of n.`
    *   **推导思路**:
        *   每个以码字 $C_k$ 为中心、半径为 $t$ 的汉明球包含的 $n$ 比特序列数量为 $V_n(t) = \sum_{i=0}^{t} \binom{n}{i}$。
        *   共有 $2^m$ 个这样的互不相交的汉明球。
        *   这些球所占据的总体积不能超过整个 $n$ 比特空间的大小 $2^n$。
    *   **球堆积界公式**:
        $$ 2^m \sum_{i=0}^{t} \binom{n}{i} \le 2^n $$
        或者改写为对 $m$ 的上界或对 $n$ 的下界：
        $$ m \le n - \log_2 \left( \sum_{i=0}^{t} \binom{n}{i} \right) $$
        $$ n \ge m + \log_2 \left( \sum_{i=0}^{t} \binom{n}{i} \right) $$
        **(板书 Part 1)**: `Sphere packing bound.`
        **(板书 Part 1)**: `[Sum_{i=0 to t} (n choose i)] * 2^m <= 2^n` (教授写的是这个形式)
    *   **意义**: 此界表明，为了达到一定的纠错能力和信息承载量，码长 $n$ 不能太短，即码率 $R=m/n$ 有一个上限。这是一个关于码性能极限的理论界限，但未给出构造方法。

2.  **上界 (Upper Bound) - 存在性界 (源自 Part 2, Part 3)**
    *   **与下界的区别**: 下界关心码性能的必要条件，上界（存在性界）关心码性能的可达性，即证明满足特定性能的码是存在的。
    *   **目标**: 给定消息长度 $m$ (即 $2^m$ 个消息) 和一个**相对最小距离 (relative minimum distance)** $\delta$ ($0 < \delta < 1/2$)，找到一个码长 $n$，使得**一定存在**一个码集 $\{C_1, \ldots, C_{2^m}\}$，其中每个 $C_i \in \{0,1\}^n$，且对所有 $i \ne j$，$d_H(C_i, C_j) \ge \delta n$。
        *   注意：$\delta n$ 对应于绝对最小距离 $d_{min}$。如果 $d_{min} \ge 2t+1$，则 $\delta \ge \frac{2t+1}{n}$。
        **(板书 Part 2)**:
        `Upper bound`
        `Given m and δ (where $0 < \delta < 1/2$ originally $t$ )`
        `$\{0,1\}^m \rightarrow \{0,1\}^n$`
        `Goal: Find n such that there must exist $C_1, C_2, \ldots, C_{2^m}$ where $C_i \in \{0,1\}^n$`
        `such that $d_H(C_i, C_j) \ge \delta \cdot n$ (bits) for all $i \ne j$.`

    *   **概率方法 (Probabilistic Method) (源自 Part 2, Part 3)**
        *   这是一种由保罗·爱多士 (Paul Erdős) 推广的非构造性证明技巧。
        *   **核心步骤**:
            1.  **随机生成码字**: (独立且均匀地) 随机选择 $2^m$ 个长度为 $n$ 的码字 $C_1, \ldots, C_{2^m}$ 从 $\{0,1\}^n$ 中。
                **(板书 Part 2)**: `1. (Uniformly) random generate $C_1 \ldots C_{2^m}$`
            2.  **估计“坏事件”概率**: 计算或估计“坏事件”发生的概率，即存在某对 $(C_i, C_j)$ 使得它们的汉明距离 $d_H(C_i, C_j) < \delta n$。
                **(板书 Part 2)**: `2. Estimate the prob $P(d_H(C_i, C_j) \ge \delta n \text{ for all } i \ne j)$`
                (教授在Part 3中改为估计坏事件概率 $P(\exists (i \ne j) \text{ s.t. } d_H(C_i, C_j) < \delta n)$)
            3.  **证明存在性**: 如果这个“坏事件”的概率严格小于1，则“好事件”（所有码对都满足距离要求）的概率严格大于0，从而证明了这种码的存在性。
                **(板书 Part 2)**: `3. If the prob > 0, done.`

        *   **引入 Chernoff Bound (切诺夫界) (源自 Part 3: 00:10)**
            *   Chernoff Bound 用于估计独立同分布随机变量之和偏离其均值的尾部概率。
            *   对于两个独立均匀随机选取的 $n$ 比特码字 $C_i, C_j$，它们的汉明距离 $d_H(C_i, C_j)$ 是 $n$ 个独立的伯努利试验（每一位是否相同）中“不同”的次数。$d_H(C_i, C_j) \sim B(n, 1/2)$，其均值为 $n/2$。
            *   我们需要估计 $P(d_H(C_i, C_j) < \delta n)$。当 $\delta < 1/2$ 时，这是对分布左尾的估计。
            *   使用 Chernoff Bound 可以得到：
                $$P(d_H(C_i, C_j) < \delta n) < e^{-n D(\delta || 1/2)} = e^{-n (\delta \log_2\frac{\delta}{1/2} + (1-\delta)\log_2\frac{1-\delta}{1/2})} = e^{-n(\delta \log_2(2\delta) + (1-\delta)\log_2(2(1-\delta)))}$$
                对于 $\delta < 1/2$，这是一个指数衰减的概率，可以简写为 $e^{-c_1(\delta)n}$ 或 $e^{-O(n)}$，其中常数依赖于 $\delta$ 与 $1/2$ 的差距。
                **(板书 Part 3)**: `Chernoff bound.`
                **(板书 Part 3)**: $P(d_H(C_i, C_j) < \delta n) < e^{-O(n)}$

        *   **联合界 (Union Bound) (源自 Part 3: 01:11)**
            *   我们关心的是是否存在 *任何一对* $(i,j)$ 距离过近。
            *   总的坏事件概率：
                $P_{bad} = P(\exists (i \ne j) \text{ s.t. } d_H(C_i, C_j) < \delta n) \le \sum_{i \ne j} P(d_H(C_i, C_j) < \delta n)$
            *   码字对的数量为 $\binom{2^m}{2} = \frac{2^m(2^m-1)}{2} < \frac{(2^m)^2}{2} = 2^{2m-1}$。
            *   所以，$P_{bad} < \binom{2^m}{2} \cdot e^{-O(n)} \approx 2^{2m-1} \cdot e^{-O(n)}$。
                **(板书 Part 3)**: $P(\exists (i \ne j) \text{ s.t. } d_H(C_i, C_j) < \delta n) < \sum_{i \ne j} P(d_H(C_i, C_j) < \delta n)$
                **(板书 Part 3)**: $< \binom{2^m}{2} \cdot e^{-O(n)}$ (教授口述约为 $2^{2m} \cdot e^{-O(n)}$)

        *   **存在性条件 (源自 Part 3)**:
            *   如果 $P_{bad} < 1$，即 $\binom{2^m}{2} \cdot e^{-O(n)} < 1$，则存在好码。
            *   取对数 (自然对数)：$\ln\left(\binom{2^m}{2}\right) - O(n) < 0$。
            *   $\ln(2^{2m-1}) - O(n) < 0 \Rightarrow (2m-1)\ln 2 < O(n) \Rightarrow 2m \ln 2 < O(n)$
            *   这意味着 $m < c_2(\delta) \cdot n$ 对于某个常数 $c_2(\delta)$。
            *   即码率 $R = m/n$ 必须小于某个常数 $c_2(\delta)$。
                **(板书 Part 3)**: $^{(i)} \binom{2^m}{2} \cdot e^{-O(n)} < 1$

    *   **Gilbert-Varshamov (GV) Bound (吉尔伯特-瓦尔沙莫夫界) (源自 Part 3: 03:30)**
        *   **(板书 Part 3)**: `Gilbert-Varshamov bound.`
        *   上述概率方法证明了：如果码率 $R = m/n$ 满足 $R < c_2(\delta)$ (一个依赖于 $\delta$ 的常数)，则一定存在一个码 $(n, 2^m)$ 其相对最小距离至少为 $\delta$。
        *   更精确的GV界（通过一种略有不同的概率方法或贪心构造法得到）表明，如果满足以下条件，则存在 $(n,M,d)$ 码：
            $$M \sum_{i=0}^{d-1} \binom{n}{i} \le 2^n$$
            对于大的 $n$，这近似于 $R \ge 1 - H_2(\delta_{GV})$，其中 $\delta_{GV} = d/n$ 是相对距离，$H_2(\cdot)$ 是二进制熵函数。
        *   教授给出的结论是定性的：`If $R < c(\delta)$ for some constant $c$ (depending on $\delta$), then $\exists C_1, \ldots, C_{2^m}$ such that $d_H(C_i, C_j) \ge \delta n$.`

---

**三、纠错码的实际考量 (源自 Part 3)**

1.  **纠错码 (ECC) 的核心任务 (源自 Part 3: 03:43)**
    1.  **设计码字 (Design codewords)**: 使得码字间汉明距离足够大以纠错。
        **(板书)**: `1) Design codewords $C_1 \ldots C_M$ such that $d_H(C_i, C_j)$ is large enough (to correct err).`
    2.  **编码 (Encode)**: 将原始消息有效地映射到码字。
        **(板书)**: `2) Encode: message $\rightarrow$ codeword`
    3.  **解码 (Decode)**: 将带噪的接收序列恢复到最可能的原始码字。
        **(板书)**: `3) Decode: received string $\xrightarrow{noise}$ codeword`

2.  **计算效率 (Computational Efficiency) 的重要性 (源自 Part 3: 04:08)**
    *   **(板书)**: `* Computational Efficiency (Encoding / Decoding)`
    *   仅仅证明“好码”的存在是不够的，还需要高效的编解码算法。
    *   **挑战**:
        *   **编码**: 若只是一个 $2^m \times n$ 的巨大查找表，当 $m$ 很大时不可行。
        *   **解码**: **最近邻解码 (Nearest Neighbor Decoding)** 虽然理论最优，但若需将接收序列与所有 $2^m$ 个码字比较 ($O(2^m \cdot n)$ 复杂度)，当 $m$ 较大时无法接受。
    *   **解决方案**: 需要码字具有**结构 (structure)**，才能设计出高效的算法。

---

**四、汉明码 (Hamming Code) 简介 (源自 Part 3: 04:40)**

*   **(板书)**: `Hamming Code`
*   汉明码是一类具有良好结构和高效编解码算法的**线性分组码 (linear block code)**。

1.  **校验矩阵 (Parity Check Matrix) $H$**
    *   以 $(7,4)$ 汉明码为例，其校验矩阵 $H$ (一个 $3 \times 7$ 矩阵)：
        **(板书)**:
        $$ H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}_{3 \times 7} $$
        (此矩阵的列是数字 1 到 7 的非零三位二进制表示的一种排列。)

2.  **伽罗瓦域 (Galois Field) $GF(2)$**
    *   汉明码（及许多线性码）定义在有限域上，通常是二元域 $GF(2) = \{0, 1\}$。
    *   $GF(2)$ 上的加法是模2加 (XOR)，乘法是模2乘 (AND)。
    *   **(板书)**: $GF(2)$ 或 $F_2$

3.  **码字由零空间 (Null Space) 定义**
    *   一个 $n$ 比特向量 $x = (x_1, \ldots, x_n)$ 是一个码字，当且仅当它满足 $H x^T = 0$ (如果 $x$ 是行向量) 或 $H c = 0$ (如果 $c$ 是列向量，且 $H$ 的列数是 $n$)。从矩阵 $H_{3 \times 7}$ 看，向量应为 $c \in \{0,1\}^7$ 列向量。
    *   码集 $C = \{ c \in \{0,1\}^7 \mid H c = 0^T \}$ (这里 $0^T$ 是 $3 \times 1$ 的零向量)。
    *   这个集合 $C$ (即 $Null(H)$) 是一个**线性子空间 (linear subspace)**。
        **(板书)**: $Null(H) = \{ x \in \{0,1\}^7 \mid H \cdot x = 0 \}$ (教授板书用 $x$，并假设 $Hx=0$)

4.  **线性子空间的维度 (Dimension)**
    *   对于一个 $r \times n$ 的校验矩阵 $H$，其零空间的维度为 $k = n - rank(H)$。
    *   对于上述 $(7,4)$ 汉明码的 $H$ 矩阵，$n=7$ (列数/码长)，$H$ 的三行线性无关 (容易验证)，所以 $rank(H)=3$ (行秩等于列秩)。
    *   因此，零空间的维度 $k = 7 - 3 = 4$。
    *   这意味着码字空间中有 $2^k = 2^4 = 16$ 个码字。这对应于 $m=k=4$ 比特的消息。
    *   所以这是一个 $(n,k) = (7,4)$ 线性分组码。

5.  **汉明码的最小距离 (引出问题)**
    *   **(板书)**: $\forall x, y \in Null(H)$, $d_H(x,y) \ge ?$
    *   对于由 $H$ 的零空间定义的汉明码，任意两个不同码字之间的最小汉明距离 $d_{min}$ 是多少？
    *   (这个问题留待下节课或作为思考题。)

---

**五、总结**

本节课程从信道编码的基本概念和动机出发，通过重复码的例子展示了通过引入冗余来提高通信可靠性的思想。接着，我们讨论了衡量编码性能的两个重要方面：**球堆积界**给出了给定纠错能力下码长的理论下限；而**概率方法**（结合Chernoff界和Union Bound）则证明了满足特定距离要求的“好码”的存在性，并引出了**吉尔伯特-瓦尔沙莫夫界**。课程强调了实际应用中**计算效率**的重要性，指出这需要码具有**结构**。最后，作为结构化码的典型代表，引入了**汉明码**，通过其**校验矩阵 $H$** 和**零空间 $Null(H)$** 来定义码字，为后续讨论其距离特性和高效编解码算法奠定了基础。

---

**六、布置的习题与思考**

1.  **Chernoff Bound 的背景知识 (源自 Part 3)**
    *   **要求**: 对于不熟悉 Chernoff Bound 的同学，需要课后自行查阅或等待教授提供的笔记。
    *   **补充说明**: Chernoff界是一类不等式，用于给出独立随机变量之和的尾部概率的上界。在本次课的应用中，两个随机码字的汉明距离可以看作是 $n$ 个独立的伯努利(参数$p=1/2$)随机变量之和（每个位置是否不同）。Chernoff界可以有效地约束这个和偏离其期望值 $n/2$ 的概率，特别是远小于期望值的情况 (如 $d_H < \delta n$ 且 $\delta < 1/2$)。

2.  **汉明码的最小距离 (源自 Part 3)**
    *   **问题**: 对于由上述 $(7,4)$ 汉明码的校验矩阵 $H$ 定义的码集 $C=Null(H)$，任意两个不同码字 $x, y \in C$ ($x \ne y$)，它们之间的汉明距离 $d_H(x,y)$ 至少是多少？即求该汉明码的最小距离 $d_{min}$。
    *   **解答思路与提示**:
        *   对于线性码，最小距离 $d_{min}$ 等于码中非零码字的最小汉明重量 $w_{min}$。即 $d_{min} = \min \{w_H(c) \mid c \in C, c \ne 0 \}$。
        *   因为如果 $x,y \in C$ 且 $x \ne y$，则 $x-y \in C$ (在 $GF(2)$ 中 $x-y = x+y$) 且 $x-y \ne 0$。而 $d_H(x,y) = w_H(x-y)$。
        *   一个非零向量 $c$ 是码字意味着 $Hc^T = 0$ (或 $Hc=0$ 若 $c$ 为列向量)。
        *   $Hc^T=0$ 意味着 $c$ 的非零位置对应的 $H$ 的列向量线性相关（在 $GF(2)$ 中，它们的和为零向量）。
        *   汉明码的校验矩阵 $H$ 的一个重要特性是：它的任何 $s$ 列都是线性无关的，但存在 $s+1$ 列是线性相关的。这个 $s+1$ 就是码的最小距离。
        *   对于标准的 $(2^r-1, 2^r-1-r)$ 汉明码（如此处的 $r=3$），其校验矩阵的列由所有非零的 $r$ 比特向量构成。
            *   任何单独一列（$s=1$）本身非零，所以 $d_{min} > 1$。
            *   任何两列（$s=2$）都是不同的，因此它们线性无关（否则 $h_i+h_j=0 \Rightarrow h_i=h_j$），所以 $d_{min} > 2$。
            *   存在三列线性相关的情况吗？例如，对于给出的 $H$ 矩阵：
                第1列 (001)$^T$, 第2列 (010)$^T$, 第3列 (011)$^T$。 $(001)^T + (010)^T + (011)^T = (0+0+0, 0+1+1, 1+0+1)^T = (0,0,0)^T$。
                是的，例如第1、2、3列是线性相关的。这意味着存在一个汉明重量为3的码字 $c=(1,1,1,0,0,0,0)$ (假设列的顺序与板书一致，但板书矩阵列是 $h_4, h_5, h_6, h_1, h_2, h_3, h_7$ 如果按标准顺序 $1, \dots, 7$ 的二进制表示)。
                如果 $H$ 的列是 $(1,2,3,4,5,6,7)$ 的二进制 $(001, 010, 011, 100, 101, 110, 111)^T$（某种顺序），例如第1列(001)$^T$，第2列(010)$^T$，第4列(100)$^T$。 $(001)^T+(010)^T+(100)^T = (111)^T \ne 0^T$。
                但是 $(001)^T + (010)^T + (011)^T = (0,0,0)^T$ (第1,2,3列)。所以，存在一个码字 $c$ 使得 $w_H(c)=3$ (其非零位对应这三列)。
        *   **结论**: 对于 $(7,4)$ 汉明码，其最小距离 $d_{min} = 3$。这意味着 $d_H(x,y) \ge 3$。因此它可以纠正 $t = \lfloor (d_{min}-1)/2 \rfloor = \lfloor (3-1)/2 \rfloor = 1$ 个比特的错误。