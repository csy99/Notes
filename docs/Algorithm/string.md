# 字符串
1.   考虑使用HashMap进行计数
2.   掌握substring, indexOf, isLetterOrDigit等常见方法
3.   考虑转换成char[]
4.   比较一定要使用equals
5.   找符合条件子串，考虑滑动窗口法，关键在于找到合法起始点
6.   KMP算法



## KMP算法

Given a string *s* and a pattern *p*, find all occurrences of *p* in *s*.

n = len(s), m = len(p)

brute force**

time: worst O(mn), average O(m+n)

**KMP**

time: worst O(m+n)

space: O(m)

```python
def match(s, p):
    n = len(s), m = len(p)
    res = []
    nxt = build(p)
    j = 0
    for i in range(n):
        while j > 0 and s[i] != p[j]:
            j = nxt[j]
        if s[i] == p[j]:
            j += 1
        if j == len(p):
            res.append(i - len(p) + 1)
            j = nxt[j]
    return res

# nxt[i]: len of the longest prefixt of p[0:i] that is also the suffix
def build(p):
    m = len(p)
    res = [0, 0]
    j = 0
    for i in range(1, m):
        while j > 0 and p[i] != p[j]:
            j = nxt[j]
        if p[i] == p[j]:
            j += 1
        res.append(j)
    return res
```

