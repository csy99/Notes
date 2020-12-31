# 实体嵌入

英文是Entity Embedding。我们希望深度学习能够处理结构化数据。在处理类别特征时，常用办法包括独热编码、顺序编码等。在NLP任务处理中，词和文章的表示都得益于嵌入。我们同样可以将这个思想在结构化数据领域加以运用。



## 原理

假设$N$表示分箱的数量，$M$表示嵌入的维度，$B$表示训练时候的批量尺寸。我们用$x \in R^{B*1}$代表输入数据，$E \in R^{N*M}$表示嵌入矩阵，$c \in R^{N*1}$表示分箱的向量中心。我们的权重。下标中$i$表示数据的序号，$j$表示分箱的序号。
$$
W_{ij} = softmax(\frac 1 {|x_i - c_j| + \epsilon})
$$
我们最后生成的嵌入矩阵可以表示为$V \in R^{B*N}$。
$$
V_i = \sum_{j=1}^N W_{ij} E_j \\\\
V = WE
$$


## 代码实现

我们借助pytorch构建实体嵌入层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityEmbeddingLayer(nn.Module):
    def __init__(self, num_level, emdedding_dim, centroid):
        super(EntityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_level, embedding_dim)
        self.centroid = torch.tensor(centroid).detach_().unsqueeze(1)
    
    def forward(self, x): 
        """
        x: size of (batch_size, 1)
        """
        eps = 1e-7
        x = x.unsqueeze(1)
        d = 1.0/((x-self.centroid).abs()+eps)
        w = F.softmax(d.squeeze(2), 1)
        v = torch.mm(w, self.embedding.weight)
        return v
```



## 适用场景

深度学习处理结构化数据。

**优点**

1. 嵌入向量稠密，非稀疏
2. 容易计算类别距离
3. 方便可视化

**缺点**

1. 暂无



# Reference

- NLP实战高手课，第三章，王然，极客时间