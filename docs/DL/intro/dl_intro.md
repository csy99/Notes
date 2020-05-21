本篇为斋藤康毅先生所编写《深度学习入门：基于python的理论和实现》中文译本的笔记。
书中所附源码可见[Github Repo]( https://github.com/hguomin/deep-learning-from-scratch)。

# 感知机

1.  接受多个输入信号，输出一个信号。信号只有0/1两种取值。

2.  神经元会计算输入信号的总和(经过加权)，当总和超过阈值(θ)时，会输出1，这也被称为神经元被激活。

3.  将阈值移到等号的另一边，行成b+w1x1+w2x2&gt;0形式的式子。b被称为**偏置**。

4.  偏置和权重的作用不相同。权重是控制输入信号的参数，偏置是调整神经元被激活的容易程度的参数。

5.  单层感知机的局限性是只能表示有一条直线分割的空间(线性空间)。

# 神经网络

1.  引入新函数h(x)改写上述公式

    y = h(b+w1x1+w2x2). h(x) = 0, if x&lt;=0; h(x) = 1, if x&gt;0.

    h(x)被称为激活函数。

2.  阶跃函数以阈值为界，一旦输入超过阈值，就切换输出。感知机中使用阶跃函数作为激活函数。

3.  最经常使用的激活函数是sigmoid函数。$h(x) = \frac{1}{1+exp(-x)}$。sigmoid函数与阶跃函数相比更加平滑。但两者都是非线性函数。

5.  ReLU函数

    h(x) = x, if x&gt;0; h(x) = 0, if x &lt;= 0

6.  softmax函数

    $y_k = \frac{exp(a_{k})}{\sum_{i = 1}^{n}{exp(a_{i})}}$

    直接按照公式实现容易出现溢出问题。改进方法：

    $y_k = \frac{C\ exp(a_{k})}{C\sum_{i = 1}^{n}{exp(a_{i})}}$ = $\frac{exp(a_k+logC)}{\sum_{i = 1}^{n}{exp(a_{i} + logC)}}$
    
    logC可以使用任意值，一般使用输入信号中最大值。

7.  **特征量**：从输入数据中准确提取本质数据的转换器。图像的特征量通常表示为向量的形式。

7. 两种针对机器学习任务的方法：

   (a) 人想到的特征量 -&gt; 机器学习(SVM/KNN)

   (b) 神经网络(深度学习)

9.  **损失函数**：神经网络寻找最优权重参数的指标。一般使用均方误差和交叉熵误差。

10.  均方误差

    $E = \frac{1}{2}\sum_{k}^{}{(y_{k} - t_{k})}^{2}$

    y~k~是神经网络的输出，t~k~是监督数据(训练数据)，k是数据维数。

11.  交叉熵误差

    $E = -\sum_{k}{t_k log({y_k})}$

    ${t_k}$是正确解标签，只有正确解标签的索引为1，其他均为0。

    使用代码实现：
	```python
    def cross_entropy_error(y, t):
		delta = 1e-7
		#log(0)是-inf，添加微小值进行保护
		return -np.sum(t*np.log(y+delta))
	```
	
12. 小批量学习(**mini-batch**)：从全部数据中选出一部分，作为全部数据的近似。

    在python中可以使用np.random.choice(train\_size, batch\_size)随机抽取。

    mini-batch版交叉熵误差代码实现：

    

    ```python
    def cross_entropy_error(y, t):
    	if y.ndim == 1:
    	t = t.reshape(1, t.size)
    	y = y.reshape(1, y.size)
    	batch_size = y.shape[0]
    	delta = 1e-7
    	return -np.sum(t*np.log(y+delta))/batch_size
    ```

    如果监督数据是标签形式(未经过独热编码)，可以将log()中的y改成`y[np.arange(batch_size), t]`。

5.  引入损失函数的原因：在神经网络的学习中，寻找最优参数时，要寻找使损失函数的值尽可能小的参数。需要计算参数的梯度，然后以次为指引，逐步更新参数的值。不能把识别精度作为指标，否则参数的梯度在绝大多数地方会变为0。调参后，损失函数可以发生连续性的变化，而识别精度是离散的值。

13. 梯度：由全部变量的偏导数汇总而成的向量称为梯度。

    $x = x - \eta\frac{\partial f}{\partial x}$

    $\eta$表示更新量，称为学习率。

    梯度下降法代码实现：

    ```python
    def grad_descent(f, init_x, lr=0.01, step=100):
    	x = init_x
    	for i in range(step):
    		grad = numerical_gradient(f, x)
    		x -= lr * grad
    	return x
    
    def numerical_gradient(f, x):
    	h = 1e-4
    	grad = np.zeros_like(x) #生成和x形状相同的数组
    	for idx in range(x.size):
    		tmp = x[idx]
    		x[idx] = tmp + h #f(x+h)
    		fxh1 = f(x)
            x[idx] = tmp - h #f(x-h)
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2)/(2*h)
            x[idx] = tmp
    	return grad
    ```

    

7.  像学习率这样的参数称为**超参数**。这是一种和神经网络的参数（权重和偏置）性质不同的参数。相对于神经网络的权重参数是通过训练数据和学习算法自动获得的，超参数则是人工设定的。一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利进行的设定。

8.  **随机梯度下降法**(SGD)：对随机选择的数据进行的梯度下降法。

9.  一个epoch表示学习中所有训练数据军备使用过一次时的更新次数。

10. 正向传播是从计算图出发点到结束点的传播。而反向传播将局部导数向正方向的反方向传递。传递这个局部导数的原理，是基于链式法则（chain rule）的。
    
19. 乘法结点：

    ```python
    class MulLayer:
        def __init__(self):
            self.x = None
            self.y = None
    
        def forward(self, x, y):
            self.x = x
            self.y = y
            return x*y
    
        def backward(self, dout):
            dx = dout * self.y
            dy = dout * self.x
            return dx, dy
    ```

    

20. 加法结点：

    ```python
    class AddLayer:
        def __init__(self):
        	pass
    
        def forward(self, x, y):
        	return x+y
    
        def backward(self, dout):
            dx = dout * 1
            dy = dout * 1
        	return dx, dy
    ```

    

20. 激活函数的ReLU层：

    $$
    \frac{\partial f}{\partial x} = \left\{ \begin{matrix}
    1,\ \  x > 0 \\
    0,\ \  x \leq 0 \\
    \end{matrix} \right.\
    $$

22. 激活函数的Sigmoid层：

    y = $\frac{1}{1 + exp( - x)}$

23. 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为放射变换。

24. Affine层：

    ```python
    class Affine:
        def __init__(self, W, b):
            self.W = W
            self.b = b
            self.x = None
            self.dW = None
            self.db = None
    
        def forward(self, x):
            self.x = x
            out = np.dot(x, self.W) + self.b
            return out
    
        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)
            return dx
    ```

    

25. 神经网络中进行的处理有推理（inference）和学习两个阶段。神经网络的推理通常不使用Softmax层。神经网络中未被正规化的输出结果有时被称为“得分”。也就是说，当神经网络的推理只需要给出一个答案的情况下，因为此时只对得分最大值感兴趣，所以不使用Softmax层。不过，神经网络的学习阶段则需要Softmax层。

26. Softmax-with-Loss层：

    ```python
    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None # 损失
            self.y = None # softmax的输出
            self.t = None # 监督数据（one-hot vector）
    
        def forward(self, x, t):
            self.t = t
            self.y = softmax(x)
            self.loss = cross_entropy_error(self.y, self.t)
            return self.loss
    
        def backward(self, dout=1):
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) / batch_size
    		return dx
    ```

    

27. **梯度确认**：指确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致的操作。数值微分的优点是实现简单，因此，一般情况下不太容易出错。而误差反向传播法的实现很复杂，容易出错。

28. 优化参数

    **SGD**低效的根本原因是梯度的方向并没有指向最小值的方向。

    **学习率衰减**：随着学习的进行，学习率逐渐减小。

    **动量法Momentum**

    v = αv - η$\frac{\partial L}{\partial W}$

    W = W + v

    代码实现：

    ```python
    class Momentum:
        def __init__(self, lr=0.01, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.v = None
    
        def update(self, params, grads):
            if self.v is None:
            	self.v = {}
            for key, val in params.items():
            	self.v[key] = np.zeros_like(val)
            for key in params.keys():
            	self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            	params[key] += self.v[key]
    ```

    

    **调整法AdaGrad**

    h = h + $\frac{\partial L}{\partial W}$$\odot \frac{\partial L}{\partial W}$

    W = W - η$\frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}$

    代码实现：

    ```python
    class AdaGrad:
        def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
    
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)
    ```

    

    **融合法Adam**

    Adam会设置3个超参数。一个是学习率（论文中以α出现），另外两个是一次momentum系数β1和二次momentum系数β2。根据论文，标准的设定值是β1为0.9，β2为0.999。设置了这些值后，大多数情况下都能顺利运行。

29. 权重的初始值

    **权值衰减**：以减小权重参数的值为目的进行学习的方法。可抑制过拟合的发生。

    权重初始值：设置较小的值，比如标准差为0.01的正态分布。

    **梯度消失**：使用的sigmoid函数是S型函数，随着输出不断地靠近0（或者靠近1），它的导数的值逐渐接近0。因此，偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。

    **表现力受限**：多个神经元输出几乎相同的值。

    **Xavier初始值**：前一层节点数越多，设定为目标节点的初始值的权重尺度越小。Xavier初始值是以激活函数是线性函数为前提而推导出来的。因为sigmoid函数和tanh函数左右对称，且中央附近可以视作线性函数，所以适合使用Xavier初始值。

    w = np.random.randn(node\_num, node\_num) / np.sqrt(node\_num)

    **He初始值**：当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是Kaiming
    He等人推荐的初始值。当前一层的节点数为n时，He初始值使用标准差为的高斯分布。

30. **Batch Normalization**算法：以进行学习时的mini-batch为单位，按mini-batch进行正规化。

    优点：可以增大学习率，不过度依赖初始值，抑制过拟合。

31. 过拟合

    产生原因：模型拥有大量参数，表现力强；训练数据少。

    **Dropout**方法：在学习的过程中随机删除神经元的方法。训练时，随机选出隐藏层的神经元，然后将其删除。被删除的神经元不再进行信号的传递。训练时，每传递一次数据，就会随机选择要删除的神经元。然后，测试时，虽然会传递所有的神经元信号，但是对于各个神经元的输出，要乘上训练时的删除比例后再输出。

    代码实现：

    ```python
    class Dropout:
        def __init__(self, dropout_ratio=0.5):
        	self.dropout_ratio = dropout_ratio
            self.mask = None
    
        def forward(self, x, train_flg=True):
            if train_flg:
            	self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            	return x * self.mask
            else:
                return x * (1.0 - self.dropout_ratio)
    
        def backward(self, dout):
            return dout * self.mask
    ```

    

    集成学习：让多个模型单独进行学习，推理时再取多个模型的输出的平均值。Dropout理解为，通过在学习过程中随机删除神经元，从而每一次都让不同的模型进行学习。并且，推理时，通过对神经元的输出乘以删除比例（比如，0.5等），可以取得模型的平均值。也就是说，可以理解成，Dropout将集成学习的效果（模拟地）通过一个网络实现了。

32. 超参数的验证

    用于调整超参数的数据，一般称为验证数据(validation data)。

    分割数据集：

    ```python
    (x_train, t_train), (x_test, t_test) = load_mnist()
    # 打乱训练数据
    x_train, t_train = shuffle_dataset(x_train,t_train)
    # 分割验证数据
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)
    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]
    ```

    

    超参数优化步骤：设定超参数的范围。从设定的超参数范围中随机采样。使用步骤1中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将epoch设置得很小）。重复步骤1和步骤2（100次等），根据它们的识别精度的结果，缩小超参数的范围。

33. 卷积神经网络(Convolutional Neural Network，**CNN**)

    之前介绍的神经网络中，相邻层的所有神经元之间都有连接，这称为**全连接**(fully-connected)。而基于CNN的网络新增了卷积层(Convolution)层和池化层(Pooling)层。

    全连接层存在的问题：数据的形状被“忽视”了。比如，输入数据是图像时，图像通常是高、长、通道方向上的3维形状。但是，向全连接层输入时，需要将3维数据拉平为1维数据。

    另外，CNN中，有时将卷积层的输入输出数据称为特征图(feature
    map)。其中，卷积层的输入数据称为输入特征图(input feature
    map)，输出数据称为输出特征图(output feature map)。

34. 卷积运算

    对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用。将各个位置上滤波器的元素和输入的对应元素相乘，然后再求和（有时将这个计算称为**乘积累加运算**）。

35. 填充

    在进行卷积层的处理之前，有时要向输入数据的周围填入固定的数据（比如0等），这称为填充(padding)。

    使用填充主要是为了调整输出的大小。比如，对大小为(4,4)的输入数据应用(3,3)的滤波器时，输出大小变为(2,2)，相当于输出大小比输入大小缩小了2个元素。这在反复进行多次卷积运算的深度网络中会成为问题。为什么呢？因为如果每次进行卷积运算都会缩小空间，那么在某个时刻输出大小就有可能变为1，导致无法再应用卷积运算。为了避免出现这样的情况，就要使用填充。

36. 步幅

    应用滤波器的位置间隔称为**步幅**(stride)。增大步幅后，输出大小会变小。而增大填充后，输出大小会变大。

    假设输入大小为(H,W)，滤波器大小为(FH,FW)，输出大小为(OH,OW)，填充为P，步幅为S。

    OH = (H+2P -FH)/S + 1

    OW = (H+2P -FW)/S + 1

    虽然只要代入值就可以计算输出大小，但是所设定的值必须使式中的和分别可以除尽。当输出大小无法除尽时（结果是小数时），需要采取报错等对策。顺便说一下，根据深度学习的框架的不同，当值无法除尽时，有时会向最接近的整数四舍五入，不进行报错而继续运行。

37. 三维数据的卷积运算

    除了高、长方向之外，还需要处理通道方向。需要注意的是，在3维数据的卷积运算中，输入数据和滤波器的通道数要设为相同的值。滤波器大小可以设定为任意值（不过，每个通道的滤波器大小要全部相同）。

    如果要在通道方向上也拥有多个卷积运算的输出，就需要用到多个滤波器（权重）。通过应用FN个滤波器，输出特征图也生成了FN个。如果将这FN个特征图汇集在一起，就得到了形状为(FN,OH,OW)的方块。将这个方块传给下一层，就是CNN的处理流。作为4维数据，滤波器的权重数据要按(output\_channel, input\_channel, height, width)的顺序书写。

38. 池化层

    池化是缩小高、长方向上的空间的运算。比如将2×2的区域集约成1个元素的处理，缩小空间大小。除了Max池化之外，还有Average池化等。相对于Max池化是从目标区域中取出最大值，Average池化则是计算目标区域的平均值。在图像识别领域，主要使用Max池化。

    特征：没有要学习的参数；通道数不发生变化；对微小的位置变化具有鲁棒性

39. 代码实现

    im2col是一个函数，将输入数据展开以适合滤波器（权重）。im2col这个名称是“image to column”的缩写，翻译过来就是“从图像到矩阵”的意思。Caffe、Chainer等深度学习框架中有名为im2col的函数，并且在卷积层的实现中，都使用了im2col。在进行卷积层的反向传播时，必须进行im2col的逆处理。池化层的实现和卷积层相同，都使用im2col展开输入数据。不过，池化的情况下，在通道方向上是独立的，这一点和卷积层不同。

40. CNN的可视化

    卷积层的滤波器会提取边缘或斑块等原始信息。而刚才实现的CNN会将这些原始信息传递给后面的层。第1层的卷积层中提取了边缘或斑块等“低级”信息，那么在堆叠了多层的CNN中，各层中又会提取什么样的信息呢？根据深度学习的可视化相关的研究，随着层次加深，提取的信息（正确地讲，是反映强烈的神经元）也越来越抽象。

41. LeNet

    它有连续的卷积层和池化层（正确地讲，是只“抽选元素”的子采样层），最后经全连接层输出结果。和“现在的CNN”相比，LeNet有几个不同点。第一个不同点在于激活函数。LeNet中使用sigmoid函数，而现在的CNN中主要使用ReLU函数。此外，原始的LeNet中使用子采样（subsampling）缩小中间数据的大小，而现在的CNN中Max池化是主流。
