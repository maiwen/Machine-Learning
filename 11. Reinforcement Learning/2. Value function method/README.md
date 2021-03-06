# 值函数法

解决无模型的马尔科夫决策问题是强化学习算法的精髓。

## 1. 蒙特卡罗方法

状态值函数和行为值函数的计算实际上是计算返回值的期望。
动态规划的方法是利用模型对该期望进行计算。在没有模型时，我们可以采用蒙特卡罗的方法计算该期望，即利用随机样本来估计期望。
在计算值函数时，蒙特卡罗方法是利用经验平均代替随机变量的期望。
此处，我们要理解两个词，何为经验？何为平均。

首先看何为经验：
![](../img/mtkljy.png)
再来看什么是平均：
这个概念很简单，平均就是求均值。不过，利用蒙特卡罗方法求状态s处的值函数时，又可以分为第一次访问蒙特卡罗方法和每次访问蒙特卡罗方法。
![](../img/mtklpj.png)

### 1.1 策略评估（递增计算均值）
![](../img/dzjz.png)
即：

![](https://img-blog.csdn.net/20180309104347140)
这种形式的更新由于不需要像传统做法那样将每个样本的值都存放在内存中，所以是一个很有效率的做法---被称为递增法（Incremental Implementation）。
### 1.2 策略改善
蒙特卡罗方法利用经验平均对策略值函数进行估计。当值函数被估计出来后，对于每个状态s ，通过最大化动作值函数，来进行策略的改进。

### 1.3 探索
由于不知道智能体与环境交互的模型，蒙特卡罗方法是利用经验平均来估计值函数。
能否得到正确的值函数，取决于经验。
**如何获得充足的经验是无模型强化学习的核心所在**。

在动态规划方法中，为了保证值函数的收敛性，算法会对状态空间中的状态进行逐个扫描。
无模型的方法充分评估策略值函数的前提是每个状态都能被访问到。
因此，在蒙特卡洛方法中必须采用一定的方法保证每个状态都能被访问到。

#### 1.3.1 探索性初始化
所谓探索性初始化是指每个状态都有一定的几率作为初始状态。
在探索性初始化中，迭代每一幕时，初始状态是随机分配的，这样可以保证迭代过程中每个状态行为对都能被选中。

探索性初始化蒙特卡罗方法：

![](http://5b0988e595225.cdn.sohucs.com/images/20180313/35402363f0244a45a5cb1379619e461a.jpeg)
#### 1.3.2 温和的探索策略
探索性初始化蕴含着一个假设，即：假设所有的动作都被无限频繁选中。对于这个假设，有时很难成立，或无法完全保证。

我们会问，如何保证初始状态不变的同时，又能保证报个状态行为对可以被访问到？答案是：精心地设计你的探索策略，以保证每个状态都能被访问到。

可是如何精心地设计探索策略？符合要求的探索策略是什么样的？
![](../img/softpolicy.png)

#### 1.3.3 同策略、异策略
根据行动策略和目标策略（策略改善公式右边策略）是否是同一个策略，蒙特卡罗方法又分为on-policy和off-policy.

##### 同策略
既然既要改进策略又需要保证探索，那么就设计一个策略（如 ε-soft策中的ε-greedy策略），
使得其有比较大的概率在任务中利用目前探索到的最优操作，而以比较小的概率随机选择一个其他操作来探索。
这样就保证了Exploration-Exploitation的平衡。
若行动策略和目标策略是同一个策略，我们称之为on-policy,可翻译为同策略。比如，行动策略和目标策略都是 ε-greedy策略。


##### 异策略
既然既需Exploration也需要Exploitation，那为什么不设计两个策略呢？一个专门负责实际探索，一个专门负责寻找最优策略。
若行动策略和目标策略是不同的策略，我们称之为off-policy,可翻译为异策略。例如用来评估和改进的策略π是贪婪策略，用于产生数据的探索性策略μ为探索性策略，如 ε-greedy策略。
用于异策略的目标策略π和行动策略μ并非任意选择的，而是必须满足一定的条件。
这个条件是覆盖性条件即：行动策略μ产生的行为覆盖或包含目标策略π产生的行为。

由于π本身并没有样本，需要从μ策略产生的样本回报来估算，所以需要通过两个策略的关系来推测---由此引入统计学中的重要性采样。

### 1.4 重要性采样

![](http://5b0988e595225.cdn.sohucs.com/images/20180313/7b1d959ebf4f4ab18a18ebdb1aa00ac0.jpeg)
重要性采样来源于求期望。
![](https://www.zhihu.com/equation?tex=%5C%5B+E%5Cleft%5Bf%5Cright%5D%3D%5Cint%7Bf%5Cleft%28z%5Cright%29p%5Cleft%28z%5Cright%29dz%7D%5C%5C%5C%5C%5Cleft%283.6%5Cright%29+%5C%5D)

当随机变量z的分布非常复杂时，无法利用解析的方法产生用于逼近期望的样本，这时，我们可以选用一个概率分布很简单，产生样本很容易的概率分布q(z) ，比如正态分布。
原来的期望可变为：

![](http://5b0988e595225.cdn.sohucs.com/images/20180313/e1a9c795d96a4df29d2dea87b463cb19.png)
![](../img/ptzyxcy.png)
基于重要性采样的积分估计为无偏估计，即估计的期望值等于真实的期望。但是，基于重要性采样的积分估计的方差无穷大。
这是因为，原来的被积函数乘上了一个重要性权重，这就改变了被积函数的形状及分布。
尽管被积函数的均值没有发生变化，但方差明显发生改变。在重要性采样中，使用的采样概率分布与原概率分布越接近，方差越小。
然而，被积函数的概率分布往往很难求得，或很奇怪，没有简单地采样概率分布能与之相似，如果使用分布差别很大的采样概率对原概率分布进行采样，方差会趋近于无穷大。

一种减小重要性采样积分方差的方法是采用加权重要性采样：
![](https://www.zhihu.com/equation?tex=%5C%5B+E%5Cleft%5Bf%5Cright%5D%5Capprox%5Csum_%7Bn%3D1%7D%5EN%7B%5Cfrac%7B%5Comega%5En%7D%7B%5CvarSigma_%7Bm%3D1%7D%5E%7BN%7D%5Comega%5Em%7D%7Df%5Cleft%28z%5En%5Cright%29%5C%5C%5C%5C+%5C%5D)

![](../img/zyxqz.png)

普通重要性采样值函数估计为:
![](https://www.zhihu.com/equation?tex=%5C%5B+V%5Cleft%28s%5Cright%29%3D%5Cfrac%7B%5CvarSigma_%7Bt%5Cin%5Cmathcal%7BT%7D%5Cleft%28s%5Cright%29%7D%5Crho_%7Bt%7D%5E%7BT%5Cleft%28t%5Cright%29%7DG_t%7D%7B%5Cleft%7C%5Cmathcal%7BT%7D%5Cleft%28s%5Cright%29%5Cright%7C%7D%5C%5C%5C%5C%5Cleft%283.11%5Cright%29+%5C%5D)

加权重要性采样值函数估计为:
![](https://www.zhihu.com/equation?tex=%5C%5B+V%5Cleft%28s%5Cright%29%3D%5Cfrac%7B%5CvarSigma_%7Bt%5Cin%5Cmathcal%7BT%7D%5Cleft%28s%5Cright%29%7D%5Crho_%7Bt%7D%5E%7BT%5Cleft%28t%5Cright%29%7DG_t%7D%7B%5CvarSigma_%7Bt%5Cin%5Cmathcal%7BT%7D%5Cleft%28s%5Cright%29%7D%5Crho_%7Bt%7D%5E%7BT%5Cleft%28t%5Cright%29%7D%7D%5C%5C%5C%5C%5Cleft%283.12%5Cright%29+%5C%5D)
由于分母不同，可见普通模式有高方差但低偏差（即样本足够真实但泛化性较差），而加权模式则相反。就实际情况而言，加权模式效果会比较好，而普通模式则极有可能因为高方差而无法收敛。

## 2. 时间差分法

时间差分(TD)方法是强化学习理论中最核心的内容，是强化学习领域最重要的成果。
与动态规划的方法和蒙特卡罗的方法比，时间差分的方法主要不同点在值函数估计上面。
相比于动态规划的方法，蒙特卡罗的方法需要等到每次试验结束，所以学习速度慢，学习效率不高。从两者的比较我们很自然地会想，
能不能借鉴动态规划中boot’strapping的方法，在不等到试验结束时就估计当前的值函数呢？

答案是肯定的，这就是时间差分方法的精髓。
时间差分方法结合了蒙特卡罗的采样方法（即做试验）和动态规划方法的bootstrapping(利用后继状态的值函数估计当前值函数)。
![](../img/td.png)
从统计学的角度来看，蒙特卡罗方法（MC方法）和时间差分方法（TD方法）都是利用样本去估计值函数的方法，哪种估计方法更好呢？
既然是统计方法，我们就可以从期望和方差两个指标对两种方法进行对比。
![](../img/mtkltd.png)

时间差分法包括同策略的Sarsa方法和异策略的Qlearning方法。
![](../img/sarsa.png)
![](../img/qlearning.png)

## 3. Deep Q-network
![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning.png)

DQN对Q-learning的修改主要体现在以下三个方面：
### 3.1 DQN利用深度卷积神经网络逼近值函数

![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning%20(1).png)

利用神经网络逼近值函数的做法在强化学习领域早就存在了，可以追溯到上个世界90年代。那时，学者们发现利用神经网络，尤其是深度神经网络去逼近值函数这件事不太靠谱，因为常常出现不稳定不收敛的情况，所以这个方向一直没有突破，直到DeepMind出现。

我们可能会问，DeepMind到底做了什么？

别忘了DeepMind的创始人Hassabis是神经科学的博士。2005年Hassabis就想如何利用人的学习过程提升游戏的智能水平，所以他去伦敦大学开始读认知神经科学的博士，很快他便做出了很突出的成就，在《science》,《nature》等顶级期刊狂发论文。他的研究方向是海马体。为什么选海马体？什么是海马体？

海马体是人大脑中主要负责记忆和学习的部分，Hassabis学认知神经科学的目的是为了提升机器的智能选海马体作为研究方向很自然。

现在我们就可以回答，DeepMind到底做了什么？

他们将认识神经科学的成果应用到了深度神经网络的训练之中！
### 3.2 DQN利用了经验回放对强化学习的学习过程进行训练

人在睡觉的时候，海马体会把一天的记忆重放给大脑皮层。利用这个启发机制，DeepMind团队的研究人员构造了一种神经网络的训练方法：经验回放。

通过经验回放为什么可以令神经网络的训练收敛且稳定？

原因是：对神经网络进行训练时，存在的假设是独立同分布。而通过强化学习采集到的数据之间存在着关联性，利用这些数据进行顺序训练，神经网络当然不稳定。经验回放可以**打破数据间的关联**。具体是这么做的：

![](https://pic3.zhimg.com/80/v2-f97a7e4542c07c326aa74fcfe1e03360_hd.jpg)

如图，在强化学习过程中，智能体将数据存储到一个数据库中，然后利用均匀随机采样的方法从数据库中抽取数据，然后利用抽取的数据对神经网络进行训练。

这种经验回放的技巧可以**打破数据之间的关联性**，该技巧在2013年的NIPS已经发布了。2015年的nature论文进一步提出了目标网络的概念进一步地减小数据间的关联性。
### 3.3 DQN独立设置了目标网络来单独处理时间差分算法中的TD偏差。

与表格型的Q-learning算法所不同的是，利用神经网络对值函数进行逼近时，值函数的更新步更新的是参数θ，更新方法是梯度下降法。因此图1.1中第6行值函数更新实际上变成了监督学习的一次更新过程，其梯度下降法为：
![](https://www.zhihu.com/equation?tex=%5C%5B%0A%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta_t%2B%5Calpha%5Cleft%5Br%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%5Cleft%28s%27%2Ca%27%3B%5Ctheta%5Cright%29-Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%5Cright%5D%5Cnabla+Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%0A%5C%5D)

我们称计算TD目标时所用的网络为TD网络。以往的神经网络逼近值函数时，计算TD目标的动作值函数所用的网络参数θ，与梯度计算中要逼近的值函数所用的网络参数相同，这样就容易使得**数据间存在关联性，训练不稳定**。为了解决这个问题，DeepMind提出计算TD目标的网络表示为θ-；计算值函数逼近的网络表示为θ ；用于动作值函数逼近的网络每一步都更新，而用于计算TD目标的网络每个固定的步数更新一次。因此值函数的更新变为：
![](https://www.zhihu.com/equation?tex=%5C%5B%0A%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta_t%2B%5Calpha%5Cleft%5Br%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%5Cleft%28s%27%2Ca%27%3B%5Ctheta%5E-%5Cright%29-Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%5Cright%5D%5Cnabla+Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%0A%5C%5D)

![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning%20(2).png)

