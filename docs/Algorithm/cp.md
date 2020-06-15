# Combination 
1. 如果需要返回的组合长度不一样，则需要在外层调用dfs时加循环

2. 注意去重

   time: O(2^n)
```java
int[] nums;
List<List<Integer>> res = new ArrayList();
int n;

// C(m, n)
private void dfs(int start, int size, List<Integer> subset) {
    if (size == subset.size()) {
        res.add(new ArrayList(subset));
        return;
    }
    for (int i = start; i < n; i++) {
        if (i > start && nums[i] == nums[i-1])  // remove duplicates
            continue;
        subset.add(nums[i]);
        dfs(i+1, size, subset);
        subset.remove(subset.size()-1);
    }
}

public List<List<Integer>> subsetsWithDup(int[] nums) {
	Arrays.sort(nums);
	this.nums = nums;
	n = nums.length;
	for (int s = 0; s <= n; s++)
		dfs(0, s, new ArrayList());
	return res;
}
```



# Permutation

1. 如果有重复数字，需要先排序，且不能使用HashSet记录是否被使用过，
   需要用一个boolean数组。    

2. 注意去重   

   time: O(n!)
```java
List<List<Integer>> res = new ArrayList();
int[] nums;
public List<List<Integer>> permute(int[] nums) {
    if (nums.length == 0) return res;
    this.nums = nums;
	Arrays.sort(this.nums);  // help remove duplicates later
    dfs(new ArrayList(), new HashSet());
    return res;
}
    
private void dfs(List<Integer> cur, HashSet<Integer> used) {
    if (cur.size() == nums.length) {
        res.add(new ArrayList(cur));
        return;
    }
    for (int i = 0; i < nums.length; i++) {
        if (used.contains(nums[i]))
            continue;
		if (i > 0 && !used[i-1] && arr[i] == arr[i-1]) // remove duplicates
			continue;
        used.add(nums[i]);
        cur.add(nums[i]);
        dfs(cur, used);
        used.remove(nums[i]);
        cur.remove(cur.size()-1);
    }
}
```


​	
​	
​	
​	
​	
​	

