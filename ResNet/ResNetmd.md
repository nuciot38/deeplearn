# 论文

- Deep Residual Learning for Image Recognition

![img](https://screenshotscdn.firefoxusercontent.com/images/fea24921-ceb7-4f00-ab67-e16cf7320a15.png)

## 优点

> 1. 相比传统的卷积神经网络如VGG复杂度降低，需要的参数下降
> 2. 可以做到更深，不会出现梯度弥散的问题
> 3. 优化简单，分类准确度加深由于使用更深的网络
> 4. 解决深层次网络的退化问题



## 原理

> ​	深度卷积残差网络是去学习输入到（输出－输入）的映射。这样就有了一个先验信息：输出一定是可以由输入的一部分组成的。事实上证明，这个先验信息是很有用的，这对应于相等映射（也就是要学习的参数都为０的情况）
>
> ​	ResNet改变的是神经网络的连接方式，将前一层（输入）直接连接到后一层（输出）,这样实现残差的学习，而参数相比传统的VGG一点都没有多，但是结构改了。



## 实现细节

>1. 图像数据上，random采样224＊224，水平翻转
>2. 采用batch normalization
>3. 使用0.0001的weight normalization
>4. 使用0.9的momentum
>5. mini_batch256张



## 具体处理

>1. 用ImageNet中的图片，随机resize[256，480]中的一个大小，然后crop成224x224. 减去均值。
>
>2. 应用论文：A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.中的标准颜色augment。
>
>3. 在每个CONV后，activation前加上BN层。
>
>4. 应用论文：K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.中的初始化方法。
>
>5. mini-batch=256， 用SGD。weight decay 0.0001，momentum=0.9 初始化学习率为0.1，遇到error plateaus就除以10。
>
>6. 共迭代600,000次。
>
>7. 不用dropout。



## 主要结论

>1. 网络足够深的情况下，ResNet效果好
>2. 网络一般深，Res能在早期阶段提供更快的优化