# 二叉树 Tree
1.	先问清楚是不是二叉树？二叉搜索树？子节点到父节点的指针？  
2.	大部分题目可以通过递归解决  
3.	掌握四种遍历树的方法inorder, preorder, postorder, level order  
4.	配合遍历的顺序，有可能需要借助额外的数据结构，比如栈  

## one root
```java
public Node solve(Node root) {
	if (root == null) 
		return null;
	Node l = solve(root.left);
	Node r = solve(root.right);
	return g(root, l, r);
}
```

## two roots
```java
public Node solve(Node p, Node q) {
	if (p == null && q == null) 
		return True;
	if (p == null || q == null) 
		return false;
	if (p.val != q.val)
		return false;
	Node l = solve(p.left, q.left);
	Node r = solve(p.right, q.right);
	return l && r;
}
```


# 前缀树 Trie
1.	终点节点的处理（这里的终点节点并不一定是叶子节点）
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

# 树状数组/二元索引树 Binary Indexed Tree
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

