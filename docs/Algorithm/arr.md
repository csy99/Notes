# 数组

1. 考虑双向遍历数组



## 快慢指针

应用：原数组上删除 

用法：实质是快慢指针，其中一个做iterator，另一个做counter



## 双向双指针
应用：原数组上rotate或者reverse，in place交换，求和

用法：前后两个指针分别向中间移动，监测left < right. 



## 建立计数器
应用：输入只包含特定数字/字母

用法：创建O(1)数组（实质上是hashtable/hashmap）



## 滑动窗口 Sliding Window

### 窗口大小固定

应用：求固定长度窗口内最大值、最小值、中位数

用法：借助额外数据结构存储，考虑HashSet、Monotonic Queue、数组等。

### 窗口大小不固定

应用：窗口内不同元素固定为k个，求满足条件窗口长度或个数

用法：和单向双指针类似，先移动右指针，不满足条件后移动左指针



## Binary Indexed Tree

见tree

