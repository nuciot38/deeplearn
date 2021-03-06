# 论文

ImageNet Classification with Deep Convolutional Neural Networks



![img](https://screenshotscdn.firefoxusercontent.com/images/f28d5fef-4c98-4023-adf6-411a3a1fb036.png)

# 结构

AlexNet为8层结构，其中前5层为卷积层，后面3层为全连接层，学习参数6千万个，神经元约有650，000个。
AlexNet在两个GPU同时训练完成。
如图所示，AlexNet第2、4、5层均是与前一层自己GPU内连接，第3层是与前面两层全连接，全连接层是2个GPU全连接。
RPN层在第1、2个卷积层后。
Max pooling层在RPN层以及第5个卷积层后。
ReLU在每个卷积层以及全连接层后。

# 数据处理

## 初步处理

首先将不同分辨率图像变换到256*256：由于ImageNet图像具有不同的分辨率，而网络要求输入的图像大小一致（由于存在全连接层），所以在数据处理中，Alexnet统一将图像变换到256*256，变换方法：首先将图像的短边缩放到256，然后在中间部分截取256*256作为训练数据。

减均值：均值采用训练数据按照RGB三分量分别求得。

## 训练数据处理

- 256*256大小的图像中，随机截取大小为227*227大小的图像

- 取镜像

- 这样使原始数据增加了（256-224）*（256-224）*2 = 2048倍

- 对RGB空间做PCA，然后对主成分做（0,0.1）的高斯扰动，结果使错误率下降1%,公式如下，其中$p_i$ ,$λ_i$分别为PCA求取的特征值，特征向量,$\alpha_i$为（0,0.1）的随机变量，对于每张图的每次训练α只计算一次。 

  $[p1,p2,p3][α1λ1,α2λ2,α3λ3]$

## 测试数据处理

- 抽取图像4个角和中心的224*224大小的图像以及其镜像翻转共10张图像利用softmax进行预测，对所有预测取平均作为最终的分类结果

# 设计

采用ReLU激活函数

> - 训练速度更快，效果更好
> - 可以不需要对数据进行normalization
> - 适用于大型网络模型的训练   

- 采用LRN层（后面网络用的不多）

        ReLU层后。
        提高泛化能力，横向抑制。
        top-1与top-5错误精度分别下降了1.4%和1.2%。
        计算公式如下：
        bix,y=aix,y/(k+α∑j=max(0,i−n/2)min(N−1,i+n/2)(ajx,y)2)β
        默认参数：(k=2,n=5,α=10−4,β=0.75)

重叠池化

    top-1与top-5 error rate 下降了0.4%与0.3%

采用Dropout

        随机关闭部分神经元，已减少过拟合
        训练迭代的次数会增加

2个GPU并行计算



# 减少过拟合方法

- Data augmentation

对抗过拟合最简单有效的办法就是扩大训练集的大小，AlexNet中使用了两种增加训练集大小的方式。

> Image translations and horizontal reflections. 对原始的256x256大小的图片随机裁剪为224x224大小，并进行随机翻转，这两种操作相当于把训练集扩大了32x32x2=2048倍。在测试时，AlexNet把输入图片与其水平翻转在四个角处与正中心共五个地方各裁剪下224x224大小的子图，即共裁剪出10个子图，均送入AlexNet中，并把10个softmax输出求平均。如果没有这些操作，AlexNet将出现严重的过拟合，使网络的深度不能达到这么深。
> Altering the intensities of the RGB channels. AlexNet对RGB通道使用了PCA（主成分分析），对每个训练图片的每个像素，提取出RGB三个通道的特征向量与特征值，对每个特征值乘以一个α，α是一个均值0.1方差服从高斯分布的随机变量。

- Dropout

> Dropout是神经网络中一种非常有效的减少过拟合的方法，对每个神经元设置一个keep_prob用来表示这个神经元被保留的概率，如果神经元没被保留，换句话说这个神经元被“dropout”了，那么这个神经元的输出将被设置为0，在残差反向传播时，传播到该神经元的值也为0，因此可以认为神经网络中不存在这个神经元；而在下次迭代中，所有神经元将会根据keep_prob被重新随机dropout。相当于每次迭代，神经网络的拓扑结构都会有所不同，这就会迫使神经网络不会过度依赖某几个神经元或者说某些特征，因此，神经元会被迫去学习更具有鲁棒性的特征。