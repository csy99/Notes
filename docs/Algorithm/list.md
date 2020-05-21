# 链表

## 概况

链表存储空间不一定连续，是临时分配的，所以不能像数组一样用index提取元素。

大量链表问题可以使用额外数据结构简化调整。但是最优解一般不使用额外数据结构。



## 解题要点

1. 搞清楚单向/双向？有环/无环？
2. 翻转链表、交换两个节点是基础
3. 头节点有可能发生变动的，考虑创建dummy head
4. 考虑新开一个链表
5. 快慢指针
6. 发生指针变动的可能需要分情况讨论
7. 把需要改变的节点存起来
8. 链表调整函数的返回值类型，一般是节点类型
9. 先画图理清思路
10.  边界条件：头尾节点、空节点
