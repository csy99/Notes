# 在cmd运行的指令

1.  list  
list all the tables

2.  !describe \`*table*\`  
return attributes related to the table

3.  create \`*table*\`, \`*column family*\`  
create a new table with a column family

4.  exit  
exit the shell

5.  curl – o \~/*file URL*  
output saved to the file with the URL

6.  head *file*  
gives a view of the file

7.  sed -i \`1d\` *file*  
get rid of the header row

8.  hadoop fs -copyFromLocal *path* *path2*

9.  hadoop fs -ls *path2*

10. \\表示下行继续  
```
hbase *mapreduceMethod* -Dimporttsv.separator=, -Dimporttsv.columns=”
HABASE\_ROW\_KEY,
*family: column*,
*family: column*,
*family: table*” *table* hdfs://*URL*```


11. scan \`*table*\`  
returns all the rows of the table

12. put ‘*table*’, ‘*row key*’, ‘*column family: column*’, ‘*value*’  
insert a new alue or update if it already exists

13. get ‘*table*’, ‘*row key*’  
returns certain row

# 在apache phoenix运行的指令

1.  语法基本与sql类似

2.  select \* from “*table*” where “*column*” = ’*value*’ 注意单双引号

3.  !outputformat vertical  
returns in vertical format

4.  !outputformat table  
returns in default format

5.  select distinct “*column*” from “*table*”;  
returns all values without repetition

6.  upsert into “*table*” select … from “*table*”  
load the data from somewhere into the table


