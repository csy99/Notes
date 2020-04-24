**General Hints**
=================

1.  Case insensitive, except folders in Linux

2.  Double quote and single quote does not matter, prefer single quote

3.  White space can concatenate different string

**Select data from tables**
===========================

1.  AS can be used to rename the chosen column, can be omitted

2.  ORDER BY is used to sort the list, ORDER BY *column name* DESC means
    in descending order

    NULL&gt;空格&gt;数字&gt;大写字母&gt;小写字母

    非字母字符出现在数字的前面或后面

    Eg: SELECT school, COUNT(1) quantity from test GROUP BY school ORDER
    BY quantity desc;

    以学校分组统计数量，降序排列

3.  Count(\*) is used to count the row return

4.  SELECT *column/\** FROM *table name* WHERE *condition* OR
    *condition* ORDER BY *column name*;

    SELECT *\** FROM *table name* WHERE *column* in (SELECT…) subselect

5.  Multi column is separated by comma

6.  -- is inline comment, /\* \*/ is multiline comment

7.  Condition: used = to judge if it is equal

8.  BETWEEN 30 AND 60 means &gt;= 30 and &lt;= 60, and smaller number
    should be the first

9.  LIKE ‘%asdf%’ match 0 or more characters as long as it contains
    ‘asdf’, \_ means there is only one character

10. IN is used to sub-select in certain columns

11. NOT should be immediately after WHERE unless it is NOT IN or NOT
    NULL

12. WHERE *column name* REGEXP ‘\^.\[a-e\].\*\$’ select all data that
    the second character is a to e, \[aeiou\] means find aeiou in the
    specified position, \[:space:\] means white space, \^ means the
    starting and \$ means the ending, . means any one character(just
    like \_), d+ means one or more d, c\* means zero or more c, p? means
    zero or one p

    \[\^..\]匹配不包含在\[\]之内的字符。a|b匹配a或者b.

13. WHERE *column name* IS (NOT) NULL works, but WHERE *column name* =
    NULL does not

14. DESCRIBE *table name* we can use DESC for short

15. SHOW CREATE TABLE *table* returns the statement of creating this
    table

16. SHOW COLUMNS FROM *table*

17. SHOW TABLE STATUS (LIKE ’*table name*’);

    括号内容无法在Navicat上编译，前半句将db中所有表的信息列出。

18. INNER/ LEFT/ RIGHT/ OUTER JOIN ON join two tables

    Eg: SELECT \* from wtest LEFT JOIN rtest ON wtest.id = rtest.id;

    怎么JOIN就是看以那张表为基础，整合后的表只显示所有基础表的信息和符号搜索条件的另一张表的信息。

    Navicat 不支持FULL OUTER JOIN.想达到此效果可使用一下语句：

    SELECT \* from wtest LEFT OUTER JOIN rtest ON wtest.id = rtest.id;

    UNION ALL

    SELECT \* from wtest RIGHT OUTER JOIN rtest ON wtest.id = rtest.id;

19. VIEW create a new view which meets certain conditions of a table

    Eg:

    CREATE VIEW designers AS

    SELECT …

    以后可以用SELECT \* FROM designers;来查看视图内容。

    好处是讲复杂查询简化为一个命令，并且隐藏读者无需看到的东西。

20. LIMIT
    一个参数表示选取多少个。两个参数时，第一个表示从哪一条记录开始（0-based），第二个表示选多少条记录。

**Create and modify tables**
============================

1.  CREATE TABLE *table name*(*variable name TYPE*);  
    Eg: 
	```sql
	CREATE table test (
		id BIGINT NOT NULL AUTO\_INCREMENT, 
		name varchar(64), 
		school varchar(64) DEFAULT ‘na’, 
		phone varchar(20),
		PRIMARY KEY(`id`), 
		UNIQUE(`phone`) 
	) CHARSET = UTF8;
	```

2.  INSERT INTO *table name* *(column name)* VALUES(*value of the
    type*); column name is not necessary

    注释：三种插入数据的方法。上面那种是最普通的，如果原来存在一条关键字相同的数据，
    将会抛出异常。第二种是INSERT
    IGNORE，若是原来存在一条关键字相同的数据，则新数据将不会被插入。第三种是REPLACE
    INTO，无论如何将会用新数据替换掉老数据。

3.  UPDATE *table name* SET *column name* = ‘*content’* WHERE
    *condition*

4.  DELETE FROM *table name* WHERE *condition*

5.  DROP TABLE (IF EXISTS) *table name* will delete the whole table from
    database, if statement is not necessary but we will get an error if
    there is no such table in the database

6.  ALTER TABLE *table name* DROP/ADD COLUMN *column name*
    给表格删除/增添列，也可以对主键等进行操作。进行ADD操作时不需要COLUMN.

7.  BEFORE *column* /AFTER *column* /FIRST/LAST 可用来确定新增列的位置

8.  ALTER TABLE *table* RENAME TO (new)*table* 重新命名表名

9.  ALTER TABLE *table* CHANGE COLUMN *column* (new)*column* (INT NOT
    NULL)重新命名列名顺便修改数据类型

10. ALTER TABLE *table* MODIFY COLUMN *column* (new)*type*只修改数据类型

11. NOT NULL initiate variables another default value rather than null

12. DEFAULT used to give default value

13. UNIQUE define variable to be the only one to have that value

14. PRIMARY KEY
    一条记录的唯一标识，主键可以是一个字段，或者多个字段联合。大概意
    思是not null + unique

15. CHECK 约束限定允许插入某列的值，与where子句语法类似

    Eg:

    CREATE TABLE BANK(

    id INT PRIMARY KEY,

    coin CHAR(1) CHECK (coin IN (‘P’, ‘N’, ‘D’))

    );

    ALTER TABLE contacts

    ADD CONSTRAINT CHECK gender IN (‘M’, ‘F’);

16. ALL, ANY, SOME

17. CREATE TEMPORARY TABLE *table*创建临时表

**Create and modify Databases**
===============================

1.  CREATE DATABASE *database name*;

2.  USE *database name*;

3.  DROP DATABASE *database name*;

4.  DEFAULT CHARSET utf8

5.  PRIMARY KEY ensure the value is unique

6.  SERIAL auto increment the number

7.  LAST\_INSERT\_ID() return the last id we entered

8.  FOREIGN KEY (*column name*) REFERENCES *column name*

**Variable Type**
=================

1.  DECIMAL(length, \# of bits after decimal point)

2.  FLOAT do not guarantee the precision

3.  CHAR(\#) is like array which always use \# bytes, while VARCHAR(\#)
    is like linked list which use up to (\#+1) bytes

4.  TIMESTAMP auto display the latest updated time

5.  ENUM use number to store a string

6.  SET can have more than one element in the list, and number can stand
    for a combination of elements

**General function**
====================

1.  GROUP BY *column name* put results in groups

2.  DISTINCT *column name* does not count duplicate value

3.  GROUP\_CONCAT(*column name* SEPARATOR ‘*separation sign*’)

4.  AVG(), MIN(), MAX(), STD(), SUM()

5.  CASE WHEN *condition1* THEN ‘value1’

    WHEN *condition2* THEN ‘value2’

    ELSE ‘value\*’

    END;

    Be aware that it will immediately execute the clause and then reach
    the end when it finds the first true condition.

6.  START TRANSACTION (at the beginning) + COMMIT (in the end) used when
    there is a long list of updates or inserts

ROLLBACK related performance will not be executed

在commit之前数据库不会发生任何改变。Rollback会将数据回滚到start
transaction 之前，以保证原子性。

存储引擎必须是BDB或者InnoDB（两种支持事务的引擎）。

改变引擎语法ALTER TABLE *table* TYPE = InnoDB;

1.  (CREATE TRIGGER *table name* AFTER INSERT ON *table name*

    FOR EACH ROW

    BEGIN

    UPDATE *table name* SET *operations* WHERE *conditions*

    END) begin, end block enables more than one statement can be
    executed, trigger will update one table automatically if another is
    updated

2.  SIGNAL SQLSTATE’\#’ SET MESSAGE\_TEXT = ‘ERROR’ throw exceptions

3.  functions of string

4.  LEFT(*column name, length*), RIGHT (*column name, length*),
    MID(*column name, starting index, length*) return according
    substring

5.  SUBSTR(*column name, start index, length*) 1-based, not end before
    but end after

6.  SELECT SUBSTRING\_INDEX (*column*, ‘*str*’, 1) FROM *table*;

    表示从该列取出指定str前所有内容。1表示寻找第一次出现的指定str

7.  UPPER(*column name*) , LOWER(*column name*), REVERSE(*column name*)
    transformation

8.  CONCAT\_WS(‘*separation sign*’, *column name*, *column name*)
    concatenate string using separation sign

9.  LPAD(*column name*, *fixed* *length*, *char*) left append a char if
    length is not enough. If length is longer than fixed length,
    truncate it to the fixed length.

10. LOCATE(‘*string*’, *column name*) find string in certain column,
    return 1 or 0.

11. drop trigger first, and then drop associated table

**Functions of numeric **
=========================

1.  int/ int return float

2.  int DIV int return int

3.  SIGN(\#) return 1 if positive, 0 if negative

4.  CONV(\# to be change, original base, new base) return \# changed
    into the new base

5.  RAND(*seed*) return random number, seed can be omitted

**Functions of numeric **
=========================

1.  NOW() return local time

2.  DAYOFMONTH(), MONTHNAME()

3.  TIME\_TO\_SEC(), SEC\_TO\_TIME()

4.  ADDTIME(), SUBTIME()

5.  DATE\_FORMAT(NOW(), ‘%Y-%m-%d %T’); standard time format

**Connection between tables**
=============================

1.  CREATE TABLE interests (newID INT PRIMARY KEY,

    Interest VARCHAR(20) NOT NULL,

    iPhone VARCHAR(20) UNIQUE,

    CONSTRAINT test\_phone\_fk //命名方式：来源表\_键字段\_fk（外键）

    FOREIGN KEY (iPhone) //外键字段名

    REFERENCES test (phone) //来源表（键字段）

    ) CHARSET = utf8;

    外键不一定必须是父表的主键，
    但必须有唯一性。创建外键需要注意四个问题。（1）关联字段的类型和长度要一致（2）关联的表编码要一致（3）删除时和更新时的设置要相同（4）某个表中是否已经有记录。

2.  自引用外键是出于其他目的而用于同一张表的主键。

3.  Junction table
    存储两个要产生关联性的表的主键，解决多对多的关系问题。

4.  关联子查询

> 需要先运行外部查询，后运行内部查询
>
> SELECT mc.first\_name firstname, mc.last\_name, mc.email email
>
> FROM my\_contacts mc
>
> WHERE NOT EXISTS
> //从my\_contacts表中选出未列入job\_current表的人的姓名和邮箱
>
> (SELECT \* FROM job\_current jc
>
> WHERE mc.contact\_id = jc.contact\_id);

1.  交叉连接：cross join

> Cross join syntax:
>
> SELECT b.boy, t.toy
>
> FROM boys AS b
>
> CROSS JOIN
>
> toys AS t;
>
> 返回两张表的每一行相乘的结果

1.  内连接：equijoin, non-equijoin, natural join

> Equijoin syntax:
>
> SELECT b.boy, t.toy
>
> FROM boys AS b
>
> INNER JOIN
>
> toys AS t
>
> ON boys.toy\_id = toys.toy\_id
>
> ORDER BY boys.boy;
>
> Natural join syntax:
>
> SELECT boys.boy, toys.toy
>
> FROM boys
>
> NATURAL JOIN
>
> toys
>
> ORDER BY boys.boy;
>
> //自然联接识别每个表中的相同名称并返回相符的记录(表中有同名列)
>
> //使用内连接时，两张表的顺序并无影响

1.  外联接: left outer join, right outer join, full outer
    join（部分RDBMS不支持）

> //外联接一定会提供数据行，如果没有找到相符的记录则在结果集中显示null

1.  自联接：self-join

> SELECT c1.name, c2.name AS boss
>
> FROM clown\_info c1
>
> INNER JOIN clown\_info c2
>
> ON c1.boss\_id = c2.id; //找出每个小丑的老板是谁
>
> //自联接能把单一表当成两张具有完全相同的信息的表来进行查询。
>
> //使用该表两次分别设定成不同的别名。

1.  联合：Union

> //把多张表的查询结果合并至一个表中。
>
> //每个select语句必须返回数量相同的列，且列的类型相同或可以互相转换。
>
> //默认清除重复值。 如果想看到重复值，使用UNION ALL
>
> //可采用新建表的方式捕获联合后数据的类型
>
> CREATE TABLE my-union AS
>
> SELECT title FROM job-1 UNION SELECT title FROM job-2;

1.  交集Intersect和差集Except

> //使用方式同上。Except返回只出现在第一个查询而不在第二个查询中的列。
>
> //MySQL中无法使用。

**Stored** **Routines**
=======================

1.  CREATE FUNCTION *function name* (*variable name variable type*)

    RETURNS *variable name variable type* // determine what to return

    RETURN … // the real thing that is returned

2.  DELIMITER *sign* reset the delimiter to the sign

    BEGIN

    …

    END *sign*

**User Authority**
==================

1.  设定根用户密码

    SET PASSWORD FOR ‘root’@’localhost’ = PASSWORD(‘*code*’);

2.  新增用户

    CREATE USER *username* IDENTIFIED BY ‘*code*’;

3.  授予权限

    GRANT *SELECT* ON *table* TO *username*;

    SELECT可替换为ALL, INSERT, DELETE,
    UPDATE等关键词。这些关键词与ON之间可插入括号，在括号内列举该表的某些列，表示用户只可对限定列进行操作。

    在TO *username* 后面加上WITH GRANT OPTION表示该用户可授权给别人。

    使用*database*.\*可把权限运用到该数据库的每一张表上。

4.  撤销权限

    REVOKE SELECT ON *table* FROM *username*;
    副作用是同时撤销该用户授权用户的权限

    REVOKE SELECT OPTION ON *table* FROM *username*;
    只撤销授予他人这一权限

    REVOKE SELECT ON *table* FROM *username* CASCADE;
    cascade表示权限的撤销具有连锁反应,通常情况下为默认值

    REVOKE SELECT ON *table* FROM *username*
    RESTRICT;若有其他用户受到影响，返回错误信息，且不执行撤销权限操作

5.  角色功能（Mysql中暂未纳入）

    CREATE ROLE *role*;创建角色

    GRANT SELECT, INSERT ON *table* TO *role*;赋予角色一些功能

    GRANT *role* TO *user*;用户将拥有角色拥有的功能

    WITH ADMIN OPTION允许用户把角色授予其他人

**Navicat for Mysql 快捷键**
============================

Ctrl+q 打开查询窗口

Ctrl+/ 注释语句

Ctrl+shift+/ 取消注释

Ctrl+r 运行语句

Ctrl+shift+r 运行选中语句

F6 打开一个命令行窗口

Ctrl+l 删除一行

Ctrl+n 打开一个新的查询窗口

Ctrl+w 关闭一个查询窗口
