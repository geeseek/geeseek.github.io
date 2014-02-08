---
layout: post
title:  "why use auc to evaluate ctr models"
date:   2014-01-08 10:25:37
categories: machine learning 
---

###为什么CTR预估使用AUC来评估模型？
ctr预估简单的解释就是预测用户点击item的概率。为什么一个回归的问题需要使用分类的方法来评估，这真是一个好问题，我尝试从下面几个关键问题去回答。

#### 1. ctr预估是特殊的回归问题

CTR预估的目标函数为
\\[ f(x) = P(+1|x) \\]
特殊之处在于目标函数的值域为[0,1]， 而且由于是条件概率，具有如下特性

\[ 
```mathjax
P(y|x)=\left\{\begin{matrix}
                                    f(x)\;\;for\,y=+1\\
                                    1-f(x)\;\;for\,y=-1
\end{matrix}\right.  
```
\]

如果将ctr预估按照一般的回归问题处理（如使用linear regression)，面临的问题是一般的linear regression的值域范围的是实数域，对于整个实数域的敏感程度是相同的，所以直接使用一般的linear regression来建立ctr预估模型很容易受到noise的影响。以Andrew Ng课程中的例子图1.b所示，增加一个噪音点后，拟合的直线马上偏移。而Logistic Regression很好的解决解决了这个问题[1]。

![Alt text](data:image,local://imgs/linear_regression_noise.gif)

#### 2. LR模型的cost function不使用平方差
一般回归问题采用的cost function是预测值和实际值的平方差，而LR模型无法采用平方差作为cost function的原因是由于基于LR模型公式的平方差函数是非凸函数。

LR模型采用的cost function是采用cross-entropy error function（也有叫做对数似然函数的），error measure是模型假设h产生训练样本D的可能性（likelihood)[2]。

假设y1=+1, y2=-1, ...yn=-1，对应的likelihood为

\[ P({x}_{1})P({y}_{1}|{x}_{1}) \bullet P({x}_{2})P({y}_{2}|{x}_{2}) ... \bullet P({x}_{n})P({y}_{n}|{x}_{n}) = P({x}_{1})f({x}_{1}) \bullet P({x}_{2})(1-f({x}_{2})) ... \bullet P({x}_{n})(1-f({x}_{n})) \]
  所以最优假设是

\\[ g = argmax(likelihood(h)) \\]
而对于LR假设存在如下特性

\\[ 1 - h(x) = h(-x) \\]
所以likelihood(h)在LR假设下可以变为

\\[ likelihood(h) = P({x}_{1})h({x}_{1}) \bullet P({x}_{2})h(-{x}_{2})...\bullet P({x}_{n})h(-{x}_{n}) \\
              = P({x}_{1})h({x}_{1}{y}_{1}) \bullet P({x}_{2})h({x}_{2}{y}_{2}) ... \bullet P({x}_{2})h({x}_{n}{y}_{n}) \\
              \propto \prod_{i=1}^{n}h({x}_{i}{y}_{i}) \\]
所以优化目标变为

\\[ {max}_{w} Likelihood(h) \propto \prod_{i=1}^{n}\theta({y}_{i}{w}^{T}{x}_{i}) \\]

\\[ \theta(t) = \frac{1}{1 + exp(-t)} \\]

 通过取负，取对数，可以将Cost Function转化为

\\[ {E}(w)={min}_{w} \frac{1}{N}\sum_{i=1}^{N}ln(1+exp(-{y}_{i}{w}^{T}{x}_{i})) \\]

我们假设有4个样本，y1=+1, y2=+1, y3=-1, y4=-1。

模型1的预测为 y1=0.9, y2=0.5, y3=0.2, y4=0.6

模型2的预测为 y1=0.1, y2=0.9, y3=0.8, y4=0.2

假如按分类问题的0/1 error来看，两个模型的效果一样的，模型1中y2和y4分类错误，模型2中y1和y3分类错误，不过这里是回归问题，按照上面cost function定义，我们可以分别计算模型1和模型2的error

模型1的error：error1 = -(ln0.9+ln0.5+ln0.2+ln0.4)/4

模型2的error：error2 = -(ln0.9+ln0.9+ln0.2+ln0.2)/4

可以看出error1 < error2, 模型1更优，实际也可以看出模型1的错误发生在分类决策面0.5附近，而模型2的错误更加“离谱”一点。

#### 3. 为什么AUC也可以用于LR模型的评估

普遍上对于AUC的认识是在分类问题中，取不同的threshold后，在横坐标为fasle positive rate，纵坐标为true positive rate平面上绘制ROC曲线的曲线下面积，所以很难理解是如何与这里的回归问题联系起来。
实际上，一个关于AUC的很有趣的性质是，它和Wilcoxon-Mann-Witney Test是等价的[3]。而Wilcoxon-Mann-Witney Test就是测试任意给一个正类样本和一个负类样本，正类样本的score有多大的概率大于负类样本的score。有了这个定义，我们就得到了另外一中计算AUC的办法：具体来说就是统计一下所有的 M×N(M为正类样本的数目，N为负类样本的数目)个正负样本对中，有多少个组中的正样本的score大于负样本的score。

仍然以上面的例子分析，y1=+1, y2=+1, y3=-1, y4=-1。

模型1的预测为 y1=0.9, y2=0.5, y3=0.2, y4=0.6

模型2的预测为 y1=0.1, y2=0.9, y3=0.8, y4=0.2

模型1： 正样本score大于负样本的pair包括(y1, y3), (y1, y4), (y2, y3)，auc为3/4=0.75

模型2： 正样本score大于负样本的pair包括(y2, y3),(y2, y4)，auc为2/4=0.5

模型1优于模型2，同上面的对数似然函数相比，这种auc的连续值同样可以评估预估概率与实际值的逼近程度，不过它强调的是预测score的偏序关系（rank）。

另外，ctr预估这种回归问题实质上也是特殊的分类问题（soft classification），求解的是分类概率。


####参考文献
[1]逻辑回归模型(Logistic Regression, LR)基础。 http://www.cnblogs.com/sparkwen/p/3441197.html

[2] Machine Learning Foundation, Coursera.

[3]AUC(Area Under roc Curve )计算及其与ROC的关系 http://www.cnblogs.com/guolei/archive/2013/05/23/3095747.html
