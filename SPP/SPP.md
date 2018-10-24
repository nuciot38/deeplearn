# 论文

- Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition



## SPP特点

> - 显著特点
>   1. 不管输入尺寸是怎样，SPP可以产生固定大小的输出
>   2. 使用多个窗口(pooling window)
>   3. SPP可以使用同一图像不同尺寸(scale)作为输入，得到同样长度的池化特征
>
> - 其他特点
>   1. 由于对输入对象的不同纵横比和不同尺寸，SPP同样可以处理，所以提高了图像的尺度不变(scale-invariance)和降低过拟合(overfitting)
>   2. 实验表明训练图像尺寸的多样性比单一尺寸的训练图像更容易使得图像收敛(convergence)
>   3. SPP对于特定的CNN网络设计和结构是独立的(只要把SPP放到最后一层卷积层的后面，对网络的结构是没有影响的，它只是替换了原来的pooling层)
>   4. 不仅可以用于图像分类而且可以用来目标检测

![img](https://screenshotscdn.firefoxusercontent.com/images/d80118bd-9c92-416c-9c28-34e6e97bd8c6.png)

- 使用多个窗口(pooling窗口，上图中蓝色，青绿，银灰的窗口，然后对feature maps进行pooling，将分别得到的结果进行合并就会得到固定长度的输出)。

## Single-size network

- 先假定固定输入图像的尺寸s=224，而此网络卷积层最后输出256层feature-maps，且每个feature-map大小为13x13(a=13)，全连接层总共256 x (9+4+1)个神经元，即全连接层输入大小为256 x (9+4+1)。即我们需要在每个feature-map得到一个数目为(f=9+4+1)的特征

## Muti-size train

- 公理：任何一个数都可以携程若干个数的平方和
  $$
  a=a_1^2+a_2^2....
  $$



- 推理：任何一个数的平方(为一个数)可以表达为若干个数的平方和
  $$
  a^2=b=a_1^2+a_2^2....
  $$



- 由于输入图像尺寸是多样的，致使我们在最后一层得到的每个feature-map大小为   a x b(a和b的大小是可变的)且feature-maps数为c，全连接层的输入为c x $f$，也就是每个feature-map要得到$f$个特征

- 由公理可得
  $$
  f=\sum_{i=1} n_i^2
  $$
  设计窗口大小为($w_1$,$w2$分别为窗口的宽和高)
  $$
  w_1=\lceil a/n \rceil
  $$

  $$
  w_2=\lceil b/n \rceil
  $$

  对应的stride($t1$,$t2$分别为水平stride和竖直stride)
  $$
  t_1=\lfloor a/n \rfloor
  $$



$$
t_2=\lfloor b/n \rfloor
$$

- 证：得到的pooling结果为$n*n$

  pooling水平移动($n1$为一行得到特征数)

  设$a=kn+p$,则$\lceil a/n \rceil=k+1$,$\lfloor a/n \rfloor=k$

  当$p>1$:
  $$
  \begin{align*}
  n1&=(a-w_1)/t_1+1\\
  &=(kn+p-\lceil a/n \rceil)/\lfloor a/n \rfloor+1\\
  &=(kn+p-(k+1))/k+1\\
  &=n
  \end{align*}
  $$

  $$
  
  $$

  当$p=0$:则$a=kn$, $\lceil a/n \rceil=\lfloor a/n \rfloor=k$
  $$
  \begin{align*}
  n2&=(a-w_1)/t_1+1\\
  &=(kn-\lceil a/n \rceil)/\lfloor a/n \rfloor+1\\
  &=(kn-k)/k+1\\
  &=n
  \end{align*}
  $$
  所以：

  $n1=n$

  pooling竖直移动($n2$为一列得到特征数)

  很容易证，竖直移动时。$n2=n$

  得证结果$n*n$

  得证pooling输出的结果固定为$f=\sum_{i=1}n_i^2$

## SPP分类

### 分类得到的结果：

1. 多窗口的pooling会提高实验的准确率
2. 输入同一图像的不同尺寸，会提高实验准确率(从尺度空间来看，提高了尺度不变性(scale invariance))
3. 用了多View(multi-view)来测试，也提高了测试结果
4. 图像输入的尺寸对实验的结果是有影响的(因为目标特征区域有大有小)
5. 因为我们替代的是网络的pooling层，对整个网络结构没有影响，所以可以使得整个网络可以正常训练。



## 与RCNN的区别

- RCNN

  > 1. 首先通过选择性搜索，对待检测的图片进行搜索出～2000个候选窗口
  > 2. 把这2k个候选窗口的图片都缩放到$227*227$，然后分别输入CNN中，每个proposal提取出一个特征向量，也就是说利用CNN对每个proposal进行提取特征向量
  > 3. 把上面每个候选窗口的对应特征向量，利用SVM算法进行分类识别。

  RCNN的计算量飞铲大，因为2k个候选窗口都要输入到CNN中，分别进行特征提取

- SPP

  > 1. 首先通过选择性搜索，对待检测的图片进行搜索出2000个候选窗口
  > 2. 特征提取阶段。把整张待检测的图片，输入CNN中，进行一次性特征提取，得到feature maps，然后在feature maps中找到各个候选框的区域，再对各个候选框采用金字塔空间池化，提取出固定长度的特征向量。(RCNN输入的是每个候选框，然后再进入CNN，因为SPP只需要一次对整张图片进行特征提取，速度会大大提升)
  > 3. 利用SVM算法进行特征向量分类识别
