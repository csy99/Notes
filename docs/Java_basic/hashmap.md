# hashmap数据结构和构造

在jdk1.8之前是数组加链表，之后是数组加链表加红黑树。

数组中存放对象Node，包含key, value, hash value, next指针。

### 参数

DEFAULT_INITIAL_CAPACITY=16，除法散列法h(k) = k mod m 要求m一般是质数。但是hashmap初始容量为2的n次幂。原因是方便‘&’运算，方便扩容之后元素的移动。

MAXIMUM_CAPACITY = 1 << 30

DEFAULT_LOAD_FACTOR = 0.75f （加载因子）经过大量统计计算得出的结论，空间和时间的平衡

TREEIFY_THRESHOLD =  8 （由链表转换成红黑树的阈值）

UNTREEIFY_THRESHOLD = 6（反树化的阈值）

MIN_TREEIFY_CAPCITY = 64 （总体Node数量达到64才能树化）

### Hash函数

采用扰动函数。高位右移和低位取异或。目的是保证hash val均匀分布。

### 方法

#### putVal

求数组下标并没有采用hash%size，而是(n-1)&hash。得到效果一模一样，效率大幅度提升。

#### resize

initialize or double table size. `new HashMap()`的时候并没有创建数组，是在putVal()中调用resize()的时候才创建。

工作包括扩容和复制。复制的时候有一段代码，判断e的哈希值与原容量的是否等于0，体现了容量为2的n次幂的好处。如果值为0，说明该元素在新数组中的下标不变，否则下标为当前下标加上原数组容量。

```java
do {
    next = e.next;
    if ((e.hash & oldCap) == 0) {
        if (loTail == null)
            loHead = e;
        else
            loTail.next = e;
        loTail = e;
    } else {
        if (hiTail == null)
            hiHead = e;
        else
            hiTail.next = e;
        hiTail = e;        
    }
} while ((e = next) != null);
```

### 死循环问题

jdk1.7的hashmap中链表采用头插法。是多线程产生的问题。



# Reference

- 你不知道的HashMap中的秘密 by 马士兵 （https://www.bilibili.com/video/BV1Ja4y1J7d7?p=3）