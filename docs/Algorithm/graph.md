# 图 Graph
1.	图形题不一定需要建图
2.	如果需要建图，考虑边是否具有方向
3.	考虑节点到本身是否有边

**建图方式**  
- 邻接矩阵 Adjacency matrix  
	
	space: O(|V|^2)  
	
	search: O(1)
	
- 邻接表 Adjacency list  
	
	space: O(|V|+|E|) = O(|V|^2) for dense graph  
	
	search: O(N)   
	
- 边表 Edge list  
	
	space: O(|E|)  
	
	search: O(N)   
	
	

## 回溯法
1.	记得还原每一步所做的改动
2.	添加结果的时候不能只是添加引用，要重新创建对象
3.	不符合要求的进行剪枝
4.	考虑输入变量是否需要排序



## 深度优先 DFS
1.	只需找到多组解其中一种的，首先考虑DFS  
2.	找环首先考虑DFS  
3. 	递归调用第一步就是判断输入是否合法  
4.  类似 preorder traversal  
5.  如果只需要一组解，或者题目保证只有一组解，那么dfs的返回类型可以是boolean，
这样能减少递归次数。如果题目要求返回所有满足条件的解，那么返回类型应当是void。  
```java
visited = [false]*|V|;

public void dfs(Node v) {
	visited[v] = true;
	for (Node child: v.neighbors)
		dfs(child);
}
```



## 广度优先 BFS

1.	找最短路径，首先考虑BFS   
2.  类似 level order traversal  
```java
public void bfs(Node v) {
	Queue q = new LinkedList();
	visited[v] = true;
	q.add(v);
	while (q.size() > 0) {
		Node cur = q.poll();
		for (Node child: cur.neighbors) {
			visited[child] = true;
			q.add(child);
		}
	}
}
```



## Spanning Tree

If tree T is a subgraph of G, and it contains all vertices of G
设Spanning Tree的顶点和边分别是V'和E'  
|V'| = |V|  
|E'| = |V| - 1, E' $\subset$ E  

### 最小生成树 Minimum Spanning Tree
1.	每一条边都有权重，使权重之后最小 
Prim's  
alg: pick a start vertex, treat it as root;  
	at each iteration, choose lowest weighted edge that connects a vertex without creating any cycles.  
	本质上是先形成一棵树，慢慢添加节点，向外扩展。   
time: O(|E|\*log|V|), binary heap + adjacency list  
```python
T = [0]
cost = 0
for _ in range(n-1):
	u, v, w = getMinEdge() # u $\in$ T, v $\notin$ T
	T.add(v)
	cost += w
return cost
```

Kruskal's  
alg: add all vertices to MST, sort edges by weights;  
	at each iteration, add edge with minimum weight that does not create a cycle.  
	pick a vertex to be root.   
	本质上是构建一棵树的一部分，最后进行连通，使之成为一棵树。  
	需要用到并查集的思想。  
time: O(|E|\*log|V|)  
```python
cost = 0
for _ in range(n-1):
	u, v, w = getMinEdge() # !connected(u,v) 
	merge(u, v)
	cost += w
return cost
```



## 拓扑排序 Topological Order

1.	BFS或者DFS都可能是解法 
```java
public void toporder(List<Node> nodes) {
	int n = nodes.size(); 
	Stack st = new Stack();
	boolean[] visited = new boolean[n];
	Node[] res = new Node[n];
	int idx = n-1;
	for (Node v: nodes) {
		if (!hasPredecessor(v)) {
			visited[v] = true;
			st.push(v);
		}
	}
	while (st.size() > 0) {
		Node cur = st.peek();
		boolean childrenVisited = true;
		Node unvisitedChild = null;
		for (Node child: cur.children) {
			if (!visited[child]) {
				unvisitedChild = child;
				childrenVisited = false;
				break;
			}
		}
		if (childrenVisited) {
			res[idx] = cur;
			idx--;
		} else {
			visited[unvisitedChild] = true;
			st.push(unvisitedChild);
		}
	}
}
```



## 并查集 Union Find

1.	找一个共同的祖先或者连接，可以使用并查集  
2. 	如果题目没有给出一个明确的n，很可能是并查集的变体  

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


