# 感知机

感知机（perceptron）是二类分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别，取+1和–1二值。
感知机对应于输入空间（特征空间）中将实例划分为正负两类的分离超平面，属于判别模型。
## 模型
假设输入空间（特征空间）是X⊆Rn，输出空间是 Y＝{+1,-1}。输入x∊X表示实例的特征向量，对应于输入空间（特征空间）的点；
输出y∊Y表示实例的类别。由输入空间到输出空间的如下函数
![](http://latex.codecogs.com/gif.latex?\\f(x) = sign(w\cdot x+b))
![](http://latex.codecogs.com/gif.latex?\\sign(x) = \left\{\begin{matrix}+1, x\geq 0
 & \\ -1, x< 0
 &
\end{matrix}\right.)
