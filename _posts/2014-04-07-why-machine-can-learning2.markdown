---
layout: post
title:  "why machine can learning2"
date:   2014-04-07 10:25:37
categories: machine learning 
---

## 为什么机器可以学习(2)？##

上一篇文章我们推导出out-of-sample error和in-sample error差距的概率上界，证明在满足一定条件下（独立随机抽样、分布相同），实际整体数据上的error基本不会偏离在样本上的error，或者说偏度的程度在概率上是可以预估的，也就是说机器在样本上学习到的模型在实际使用中的效果是可以预估的。 

$$
P[\left |E_{in}(g) - E_{out}(g)\right | > \varepsilon ] \leq \sum_{m=1}^{M}P[\left |E_{in}(h_{m}) - E_{out}(h_{m}) \right |] \leq 2Mexp(-2\varepsilon ^2 N)
$$

同时我们也留下一些没有解决的问题，如：

### M为无穷大怎么办？ ###
一个简单的模型，如线性模型 ax + b，参数a，b能够产生的函数（假设）个数为无穷大，这时候误差的边界就失去了意义。
为了后续的推导，我们定义概率边界
$$
    \delta  = 2Me^{-2\epsilon^{2}N}
$$
从而
$$
    \epsilon = \sqrt{\frac{1}{2N}ln(\frac{2M}{\delta})}
$$

回到如何解决M为无穷大的问题上来，假设我们面对的是之前讨论过的0/1分类问题，虽然假设空间中候选假设的数目M为无穷大，但对于有限个样本来说，预测的结果就是有限的，例如对于假设集合sign（ax +b)， 虽然a和b的取值有无线多种可能，对于样本只有A，B的情况下，任何一个假设预测的结果只能是（＋1，＋1），（＋1，－1），（－1，＋1），（－1，－1）四种情况中的一种。所以，假设样本的数量是N，则必然有

$$
    M \leq 2^{N}
$$

好，到这里我们可以将概率边界从无穷大变为有限值，很明显这个边界仍然很宽松，我们可以可以再缩小一下M的范围。

我们来看一个更理想的模型，假设h(x) = sign(x-a) ，就是当x > a时，预测分类为+1, 反之为-1，假设有N个样本（对于数轴上N个点），我们发现对应的预测结果组合只有N + 1种，所以不同的模型对应的M的上界不同，我们也许能找到M更加严格的上界。

我们定义一个假设集合的growth function如下：
$$
    M \leq m_{H}(N) = MAX_{x_{1},...,x_{N}}|H(x_{1},...,x_{n})| \leq 2^{N}
$$
就是假设集合在给定N个样本上能产生的最多的不同预测组合。
对于上面的模型假设h(x) = sign(x-a)，
$$
    m_{H}(N) = N + 1
$$
十分遗憾的是，对于不同的模型，要找出准确的growth function很难，相反的我们通过寻找一个k值来估计growth function， k满足如下条件
$$
   m_{H}(k) < 2^{k}
$$
如果存在这样的k，我们把k叫做假设空间的break point。仍然以线性模型sign(ax + b)为例，k = 4，也就是说对于任意4个样本，任何sign(ax+b) 给出的不同预测结果小于16种。
实际上，如果对于假设空间有break point上存在的话，数学上可以证明（证明可以参考[1])
$$
   m_{H}(N) \leq \sum_{i=0}^{k-1}\binom{N}{i}
$$

为了后续更好的讨论，我们定义一个假设空间的VC-Dimension(The Vapnik-Chervonenkis dimension)为
$$
    d_{vc} = k - 1
$$
所以有
$$
    m_{H}(N) \leq \sum_{i=0}^{d_{vc}}\binom{N}{i}
$$
对于不等式的右边，通过归纳法可以证明
$$
m_{H}(N) \leq \sum_{i=0}^{d_{vc}}\binom{N}{i} \leq N^{d_{vc}} + 1
$$
嗯，现在已经将假设的空间缩小到N的vc-dimension次多项式级别了。一般说来，采用线性决策面的模型vc-dimension为r+1（r为变量维度）。

### 样本数量N要多少才算够？ ###
上面讨论的是给定N个样本集合，训练出来的模型在实际使用中误差边界。
另外一个有趣的问题是，对于一个模型而言，需要多大的样本数量才能保证在实际使用中的效果？

我们先回顾上一节的成果，
$$
P[\left |E_{in}(g) - E_{out}(g)\right | > \varepsilon ]  \leq 2Mexp(-2\varepsilon ^2 N)
$$
对于下面的误差容忍度
$$
    \delta  = 2Me^{-2\epsilon^{2}N}
$$
误差的范围如下（比较常见的有0.05，即允许5%的可能in-sample和out-of-sample error超出误差范围）
$$
    \epsilon = \sqrt{\frac{1}{2N}ln(\frac{2M}{\delta})}
$$

对应的保证有1-0.05=0.95, 95%的概率
$$
    E_{out}(g) \leq E_{in}(g) + \epsilon \leq E_{in}(g) + \sqrt{\frac{1}{2N}ln(\frac{2M}{\delta})} \leq E_{in}(g) +\sqrt{\frac{1}{2N}ln(\frac{2m_{H}(N)}{\delta})}
$$

我们把这个out of sample error的概率上界叫做VC generalization bound，实际上，可以通过较为复杂的数学证明得到更为确切的边界
$$
    E_{out}(g) \leq E_{in}(g) + \epsilon \leq E_{in}(g) +\sqrt{\frac{8}{N}ln(\frac{4m_{H}(2N)}{\delta})}
$$

那么回到本节中的问题，假设我们有一个误差容忍度（如0.05）和确定的假设空间（growth function确定），N取多少才够？ 由上面的式子我们知道，N满足下面的不等式时，我们可以保证模型在实际中的效果
$$
  \sqrt{\frac{8}{N}ln(\frac{4m_{H}(2N)}{\delta})} \leq \epsilon
$$
此时边界范围足够的小。对应N的表达式满足
$$
    N \geq \frac{8}{\epsilon^{2}}ln(\frac{4((2N)^{d_{vc}}+1)}{\delta})
$$

虽然不等式左右都含有N，但我们已经可以通过递归地尝试取不同的N了。假设我们的模型假设是线性模型sign(ax+b)， 错误范围和容忍度都为0.1， 我们先假设N=1000，通过不等式计算如下

$$
    N \geq \frac{8}{0.1^{2}}ln(\frac{4((2N)^{3}+1)}{0.1}) \approx 21193
$$
 我们将N=21193代入继续迭代，最终可以得到收敛的N大致为30000左右，即符合条件的合适的样本数量.

当错误范围核容忍度都取0.1时，数学上可以证明 
$$
N \approx 10,000d_{vc}
$$
实际工作中上，N的经验取值为
$$
N \approx 10d_{vc}
$$

###参考文献###
[1] Learning From Data-A Short Course, p48
