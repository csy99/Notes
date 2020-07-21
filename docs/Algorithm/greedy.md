# 贪心算法

1. 排序！如果是内置数据结构，直接使用自定义排序。排序之后考虑能否使用binary search。

```java
Arrays.sort(arr, new Comparator<int[]>() {
    public int compare(int[] a, int[] b) {
        return Integer.compare(b[0], a[0]);
    }
});
```

