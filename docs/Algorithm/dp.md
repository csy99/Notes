# 动态规划 Dynamic Programming
1.	检查能不能拆分成子问题，且是否会被求解多次  
2.	四大要素：状态，初始化，状态转移方程，结果  
3.	优化空间，规律交错的题目尝试声明两个变量来存（一维DP两种状态）  
4.	优化时间主要看内层循环的起始位置  
5.	不一定会根据index做dp的索引，有可能是根据value。比如背包问题，dp\[n]\[W]。
内层循环w时候，不需要从0开始，可以从w\[i]开始即可。  

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

