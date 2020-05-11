# 并查集 Union Find
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


