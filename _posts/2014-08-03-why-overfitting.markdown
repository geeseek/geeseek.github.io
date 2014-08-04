---
layout: post
title:  "why overfitting(3)"
date:   2014-08-03 11:56:37
categories: machine learning 
---


##为什么会存在过拟合（overfitting）问题？##

###过拟合是如何发生的？
过拟合可能是机器学习领域中经常听到的名词，譬如大家经常说决策树模型(decision tree)容易过拟合，逻辑回归模型（LR)不容易过拟合，直觉上大家认为过拟合是因为使用过于复杂的模型拟合训练数据造成的，实际上并非如此简单。
假设有如下两个目标函数（10次和50次多项式函数）及对应的数据

考虑下面两个有趣的问题

(1) 对于10次多项式函数产生的数据，如果我们分别使用2次和10次多项式函数来拟合（学习），哪个效果更好？ （此处思考答案1min）

![Alt text](http://ww1.sinaimg.cn/bmiddle/005I3BEujw1ej0s8vd12sj30vk0rmgr7.jpg)


下面是基于图中数据训练的结果，
![Alt text](http://ww3.sinaimg.cn/bmiddle/005I3BEujw1ej0sa7lbn6j30vk0sqjxn.jpg)

|       |     2nd Order function |   10nd Order function   |
| :-------- | --------:| :------: |
| E_in    |   0.050 |  0.034  |
| E_out    |   0.127 |  9.00  |

使用10次多项式函数来拟合在验证集上效果居然远不如使用2次多项式函数。
why？ 从上面面的图你可以看到训练出来的10次函数模型并不能完全拟合所有点，那些无法拟合的点在实际中被称为噪音（noise）。10次函数模型实际上因为拟合噪音点而导致在验证集上效果很差，而2次函数模型拟合的是数据的大致趋势，所以相对在验证集上相对较好。

所以，**噪音是导致模型过拟合的原因之一（越是复杂、拟合能力强的模型越容易在这种情况造成过拟合）。**

（2） 假如没有噪音呢？ 考虑下面50次多项式函数产生的数据，如果我们分别使用2次和10次多项式来拟合，结果会有变化吗？

![Alt text](http://ww2.sinaimg.cn/bmiddle/005I3BEujw1ej0sactpoej30vk0qujwn.jpg)

下面是基于图中数据训练的结果，
![Alt text](http://ww2.sinaimg.cn/bmiddle/005I3BEujw1ej0sai92q1j30vk0sq44r.jpg)


|       |     2nd Order function |   10nd Order function   |
| :-------- | --------:| :------: |
| E_in    |   0.029 |  0.000001  |
| E_out    |   0.120 |  7680  |

why？ 因为实际目标函数（50次多项式）远比10次多项式复杂，对于2次和10多项式模型来说，比如有一些数据点无法拟合（实际上这些点称为deterministic noise)，这样对于极为有限的15个数据点，更复杂、拟合能力更强的10次多项式模型“自以为完美”但实际上悲剧的拟合了这些数据点。所以，**实际目标函数的复杂性过大而训练数据相对过小是导致过拟合的另外一个重要原因。**

**总而言之，实际目标函数的复杂度与样本数据的质量、数量决定了是否产生过拟合。**


###如何解决过拟合问题？

####正则化(regularization)
从上一节来看我们的结论是尝试使用简单的模型在训练数据上得到和复杂模型同样的效果，实际上多项式模型我们可以统一以下面的公式来表达：

$$
H(Q) = \sum_{q=0}^{Q}\omega_{q}x_{q}
$$

一种比较笨的方法是限制多项式的次数（hard order constraint）来降低模型复杂度，如

$$
\omega_{q} = 0 \; for \; q > 2
$$

带来的一个问题是，如何在次数固定的情况下避免过拟合呢？以使用直线（Q=1）拟合只有两个数据点正弦函数为例

$$
f(x) = sin(\pi x)
$$

![Alt text](http://ww4.sinaimg.cn/bmiddle/005I3BEujw1ej0sapb4stj30vk0qegs2.jpg) 

图5:f(x)上任意两点使用直线拟合的情况

![Alt text](http://ww2.sinaimg.cn/bmiddle/005I3BEujw1ej0sau3cxlj30vk0nqag8.jpg)

图6: 我们希望能够拟合的直线

可以看到，如果以常见的平方差为损失函数，图6明显去掉了一些斜率很大的直线（即|w1|很大），这些曲线使用f(x)的其他数据点来评测时会导致很大的Out of Sample Error。

从上面的例子，我们可以改进一下模型的限制，如下所示

$$
\sum_{q=0}^{Q}\omega_{q}^{2} \leq C
$$

通过C的设置，可以避免上面例子中|w1|过大的情况。
当然你也可以将限制变为如下的样子(low-order regularizer)

$$
\sum_{q=0}^{Q}q\omega_{q}^{2} \leq C
$$

这样可以同时考虑多项式的次数和同次数下参数的选择。

加入regularization后，我们的优化目标变成：

$$
min\;E_{in}(\omega)  \; subject \; to \; \omega^{T}\omega \leq C
$$

这里有两个问题：一是带约束的优化问题不太好求解，而是这样实际上已经改变了可选择的参数范围，模型的复杂度降低(VC-bound)。实际上，经过数学上的证明（较为繁琐、此次略去），可以将上面的优化目标转化为下面等价的无约束的优化目标：

$$
min\;E_{in}(w) + \lambda_{C}\omega^{T}\omega 
$$

即大家常见的regularization方式，值得注意的是这种方式没有改变模型的vc-dimension，而是改变了learning的的算法。

为什么regularization可以缓解overfitting的问题？回顾VC-bound理论：

$$
E_{out}(h) \leq E_{in}(h) + \Omega(h)
$$

regulaization实际上是保证第一部分不变的情况下，降低了第二部分的值（模型的复杂度）。

####验证(validation)
关于regularization其实有一个问题没有解决掉，就是如何选择lambda参数，实际问题中我们往往会选择一些经验值来尝试，那如何确定哪个lambda能够在测试集中表现更好呢？
我们仍然来凭直觉来看看：
(1) 直接使用看哪个lambda在训练集上的效果最好，就使用那个模型。
想想看，如果使用20次多项式去拟合10次多项式函数产生的数据（有噪音），那么lambda=0时，在训练集上效果最好，这样我们仍然得到了一个过拟合的模型。

(2) 直接在测试集上比较，哪个模型好就用哪个。
这是最容易犯的一个错误，本质上这是将测试数据当做另外一份训练数据，基于规则来训练lambda，就不再满足hoeffdin 不等式的条件，这时候得到的Error是in sample error而不是out of sample error。

在机器学习实践中我们常使用的是称为(验证)validation的技术，这个技术简单说来就是从训练集中分出一部分数据来预估out of sample error，避免出现in sample error很低，out of sample error很高的问题(过拟合问题)。

假设validation从训练集中取K条数据作为验证集，那么在验证集上的Error可以计算如下：

$$
    E_{val}(g^{-}) = \frac{1}{K}\sum_{x_{n}\in D_{val}}e(g^{-}(x_{n}),y_{n})
$$

$$
g^{-}:在去除K条数据后的训练集上找到的最优假设 
$$

e为损失函数，如平方差函数。为什么validation上的Error可以用来预估out of sample error,回顾一下hoeffdin不定式关于in sample error和out of sample的关系：

$$
   E_{out} \leq E_{in} + O(\sqrt{\frac{1}{N} ln(\frac{d_{vc}}{\sigma}}))
$$

在validation这个例子中，只有一个候选假设(如果多个假设，则是model selection问题), 样本数N为K，在误差范围指定的情况下， 有：

$$
   E_{out}(g^{-}) \leq E_{val}(g^{-}) + O(\frac{1}{\sqrt{K}})
$$

我们不加证明的认为（原则上训练数据加大产生的模型更好）

$$
   E_{out}(g) \leq E_{out}(g^{-})
$$

所以最终

$$
  E_{out}(g) \leq E_{val}(g^{-}) + O(\frac{1}{\sqrt{K}})
$$

所以，validation之所以可以用来估计out of sample error的本质原因是因为它的模型复杂度(只有一个候选假设)远远低于模型训练过程中的模型假设空间的复杂度。
此外，validation的一个重要作用是model selection（不同的model可以使lambda的选择不同，也可是percentron和LR这种模型假设上的不同），假设需要在M个模型中选择，那么则有：

$$
  E_{out}(g) \leq E_{val}(g^{-}) + O(\sqrt{\frac{lnM}{K}})
$$

由于M一般也远小于训练过程中的模型假设复杂度， 所以validation error最小的模型相应的out of sample error也更小。

####交叉验证(cross validation)
在validation集合构造时，我们需要选择合适的K来构造验证集。因为根据

$$
  E_{out}(g) \leq E_{out}(g^{-}) \leq  E_{val}(g^{-}) + O(\sqrt{\frac{lnM}{K}})
$$

根据后一个不等式，K越大，对于out of sample error误差的估计就越小（上界越低）。是不是真的K越大越好？
当然不是，实际上，K的大小也会影响第一个不等式，K越小（如k=1），

$$
   E_{out}(g) 和 E_{out}(g^{-})  越接近
$$

所以，K的选择似乎陷入一个两难的境地。
为了解决这个问题，实际中我们使用cross validation来预估out of sample error，假设K=1，但我们计算validation error时，分别计算每个样本作为为validation数据集的error的平均值。

$$
E_{CV} = \frac{1}{N} \sum_{n=1}^{N}e_{n}
$$

在数学上并不能很严格地证明cross validation能够保证获得非常低的误差上界，然而在实践中被证明比单次的validation更有效地预估out of sample error.同时，如果K=1，有M个候选模型，那么计算量会达到O(M*N)的级别，在实践中，处于性能的考虑，常常将数据随机等比例切分为V份，然后分别使用1份来作为validation集合，这种方法成为v-fold cross validation，实际中v的取值范围为5~10。




