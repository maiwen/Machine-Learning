# 策略梯度法

在值函数的方法中，我们迭代计算的是值函数，然后根据值函数对策略进行改进；
而在策略搜索方法中，我们直接对策略进行迭代计算，也就是迭代更新参数值，直到累积回报的期望最大，此时的参数所对应的策略为最优策略。

策略梯度法相比于值函数方法的优缺点：

优：

* 直接策略搜索方法是对策略π进行参数化表示，与值函数方中对值函数进行参数化表示相比，策略参数化更简单，有更好的收敛性。
* 当要解决的问题动作空间很大或者动作为连续集时，值函数方法无法有效求解。
* 直接策略搜索方法经常采用的随机策略，可以将探索直接集成到策略之中。

缺：

* 策略梯度法容易收敛到局部最小值。
* 评估单个策略时并不充分，方差较大。

## 1. 策略梯度推导

### 基于似然率来推导策略梯度
![](../img/pgtd.png)
这时，策略搜索方法，实际上变成了一个优化问题。解决优化问题有很多种方法，比如：最速下降法，牛顿法，内点法等等。

其中最简单，也是最常用的是最速下降法，此处称为策略梯度的方法。
![](../img/pgtidu.png)
我们可以利用经验平均来进行估算:

![](https://www.zhihu.com/equation?tex=%5C%5B+%5Cnabla_%7B%5Ctheta%7DU%5Cleft%28%5Ctheta%5Cright%29%5Capprox%5Chat%7Bg%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%7B%5Cnabla_%7B%5Ctheta%7D%5Clog+P%5Cleft%28%5Ctau+%3B%5Ctheta%5Cright%29R%5Cleft%28%5Ctau%5Cright%29%7D+%5C%5D)

### 从重要性采样的角度推导策略梯度
![](../img/zytd1.png)
![](../img/zytd2.png)

### 似然率策略梯度的直观理解
前面我们已经利用似然率的方法推导得到了策略梯度公式。
![](../img/srltdlj.png)

### 求似然率的梯度
![](../img/srltd.png)
似然率梯度转化为动作策略的梯度，与动力学无关。

由此，我们已经推导出了策略梯度的计算公式：

![](https://www.zhihu.com/equation?tex=%5C%5B+%5Cnabla_%7B%5Ctheta%7DU%5Cleft%28%5Ctheta%5Cright%29%5Capprox%5Chat%7Bg%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%7B%5Cleft%28%5Csum_%7Bt%3D0%7D%5EH%7B%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi_%7B%5Ctheta%7D%5Cleft%28u_%7Bt%7D%5E%7B%5Cleft%28i%5Cright%29%7D%7Cs_%7Bt%7D%5E%7B%5Cleft%28i%5Cright%29%7D%5Cright%29%7DR%5Cleft%28%5Ctau%5E%7B%5Cleft%28i%5Cright%29%7D%5Cright%29%5Cright%29%7D+%5C%5D)


（6.8）式给出的策略梯度是无偏的，但是方差很大。我们在回报中引入常数基线b来减小方差。

![](https://www.zhihu.com/equation?tex=%5C%5B+%5Cnabla_%7B%5Ctheta%7DU%5Cleft%28%5Ctheta%5Cright%29%5Capprox%5Chat%7Bg%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%7B%5Cnabla_%7B%5Ctheta%7D%5Clog+P%5Cleft%28%5Ctau%5E%7B%5Cleft%28i%5Cright%29%7D%3B%5Ctheta%5Cright%29R%5Cleft%28%5Ctau%5E%7B%5Cleft%28i%5Cright%29%7D%5Cright%29%7D+%5C%5C+%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%7B%5Cnabla_%7B%5Ctheta%7D%5Clog+P%5Cleft%28%5Ctau%5E%7B%5Cleft%28i%5Cright%29%7D%3B%5Ctheta%5Cright%29%5Cleft%28R%5Cleft%28%5Ctau%5E%7B%5Cleft%28i%5Cright%29%7D%5Cright%29-b%5Cright%29%7D+%5C%5D)

### 行动者-评论家（actor-critic）体系
上式取基线b为当前状态值函数，则为AC方法。
![](../img/ac.png)