谓词下推(predicate pushdown) 
基本策略是，始终将过滤表达式尽可能移至靠近数据源的位置。  
在传统关系型数据库中，优化关系 SQL 查询的一项基本技术是，
将外层查询块的 WHERE 子句中的谓词移入所包含的较低层查询块（例如视图），
从而能够提早进行数据过滤以及有可能更好地利用索引。  
大白话解释：在进行join等后续操作之前，将数据规模减小。  

# 优化方式
- 由谁来完成数据过滤    
- 指何时完成数据过滤  

优点： 
1. 大多数表存储行为都是列存，列之间独立存储，扫描过滤只需要扫描join列数据（而不是所有列），
如果某一列被过滤掉了，其他对应的同一行的列就不需要扫描了，这样减少IO扫描次数。  
2. 减少了数据从存储层通过socket(甚至TPC）发送到计算层的开销。    
3. 减少了最终hash join执行的开销。  
 
# 逻辑执行计划优化层面
比如SQL语句：
```sql
select * from order, item 
where item.id = order.item_id and item.category = ‘book’
```
正常情况语法解析之后应该是先执行Join操作，再执行Filter操作。
通过谓词下推，可以将Filter操作下推到Join操作之前执行。
即将where item.category = ‘book’下推到 item.id = order.item_id之前先行执行。  
# 实现层面
谓词下推是将过滤条件从计算进程下推到存储进程先行执行，注意这里有两种类型进程：
计算进程以及存储进程。
计算与存储分离思想，这在大数据领域相当常见，
比如最常见的计算进程有SparkSQL、Hive、impala等，负责SQL解析优化、数据计算聚合等，
存储进程有HDFS（DataNode）、Kudu、HBase，负责数据存储。
正常情况下应该是将所有数据从存储进程加载到计算进程，再进行过滤计算。
谓词下推是说将一些过滤条件下推到存储进程，直接让存储进程将数据过滤掉。
这样的好处是过滤的越早，数据量越少，序列化开销、网络开销、计算开销这一系列都会减少，
性能自然会提高。  



# Reference
- https://blog.csdn.net/baichoufei90/article/details/85264100
- https://juejin.im/post/5daff8edf265da5b905f0362
- http://hbasefly.com/2017/04/10/bigdata-join-2/
