# 数学
1.   考虑int溢出，除数为0，数字不能以0开头（0除外）等特殊情况。
2.   考虑gcd，lcm
3.   考虑二分搜索



## Fast Power

$x =  a * b^n$, n is a positive integer, a and b can be integers or matrices

algorithm: fast power by squaring

time: O(log n)

```python
def power(a, b, n):
    x = a
    while n > 0:
        if n & 1 == 1:
            x = x * b
        b = b * b
        n = n >> 1
    return x
```



## 随机
-    善用rand.nextInt()，实质在考察概率



## 位运算

-    找不同或出现奇数次的数，一定会用到异或
-    注意java中右移和不带符号右移