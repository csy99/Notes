# Union Find
```java
class UFS {  
    int[] parents;  
    int[] ranks;  
      
    public UFS(int n) {  
        parents = new int[n+1];  
        ranks = new int[n+1];  
        for (int i = 0; i < n+1; i++) {  
            parents[i] = i;  
            ranks[i] = 1;  
        }  
    }  
      
    public int find(int u) {  
        while (u != parents[u]) {  
            parents[u] = parents[parents[u]];  
            u = parents[u];  
        }  
        return u;  
    }  
      
    public boolean union(int u, int v) {  
        int ru = find(u);  
        int rv = find(v);  
        if (ru == rv) return false;  
          
        if (ranks[ru] > ranks[rv])  
            parents[rv] = ru;  
        else if (ranks[rv] > ranks[ru])  
            parents[ru] = rv;  
        else {  
            parents[ru] = rv;  
            ranks[rv] += 1;  
        }   
        return true;      
    }  
}  
```

# Binary Indexed Tree
``` java
class BIT {
    int[] sum;
    public BIT(int n) {
        sum = new int[n+1];
    }
    
    public void update(int i, int diff) {
        while (i < sum.length) {
            sum[i] += diff;
            i += i & (-i);
        }
    }
    
    public int query(int i) {
        int res = 0;
        while (i > 0) {
            res += sum[i];
            i -= i & (-i);
        }
        return res;
    }
}
```

# Dynamic Programming

## depend on constant number of its sub-problems  
```java
// T(n), S(n)->S(1)
dp = new int[n]
for i = 1 to n: 
	dp[i] = f(dp[i-1], dp[i-2], ...)
return dp[n]
```
## depend on all its sub-problems
```java
// T(n^2), S(n)
dp = new int[n]
for i = 1 to n:  // problem size
	for j = 1 to i - 1:  // sub-problem size
		dp[i] = max(dp[i], f(dp[j]))
return dp[n]
```
## two inputs
```java
// T(mn), S(mn)
dp = new int[m][n]
for i = 1 to m:
	for j = 1 to n:
		dp[i][j] = f(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
return dp[m][n]
```
## depend on uncertain sub-problems
```java
// T(n^3), S(n^2)
dp = new int[m][n]
for l = 1 to n:  // problem size
	for i = 1 to (n-l):  // sub-problem start point
		j = i+l-1  // end point
		for k = i to j:
			dp[i][j] = max(dp[i][j], f(dp[i][k], dp[k][j]))
return dp[1][n]
```
## input has two dimensions
Personally, this is very similar to two inputs.   
```java
// T(mn), S(mn)
dp = new int[m+1][n+1]
for i = 1 to m:
	for j = 1 to n:
		dp[i][j] = f(dp[i-1][j], dp[i][j-1])
return dp[m][n]
```
## input has two dimensions, depends on some sub-problems
```java
// T(kmn), S(kmn)->S(mn)
dp = new int[m][n]
for k = 1 to K:
	for i = 1 to m:
		for j = 1 to n:
			dp[k][i][j] = f(dp[k-1][i+di][j+dj])
return dp[K][m][n]
```

# Trie
```java
class TrieNode {  
    TrieNode[] children;  
    boolean isWord;  
    public TrieNode() {  
        children = new TrieNode[26];  
        isWord = false;  
    }  
}  
```

