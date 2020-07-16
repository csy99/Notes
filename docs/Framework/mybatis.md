# JDBC

缺点：

- 非对象，修改表需要重写api。没用连接池，操作数据库需要频繁的创建和关联链接
- 修改sql的话需要重编译java，不利于系统维护
- 使用PreparedStatement预编译对变量进行编号，序号不利于维护
- 返回结果集需要硬编码



# MyBatis

### 宏观

连接数据库。

数据源：Driver, URL, username, password

执行语句：CRUD

操作：Connection, PreparedStatement, ResultSet

### 全局配置

mybatis_config.xml

功能：分页，监控，日志，记录sql，数据埋点，逆向工程

```xml
<plugin interceptor="SqlPrint"> </plugin>
```

### 获取数据库源

XMLConfigBuilder中`parseConfiguration()` 解析mybatis全局配置标签。`environmentsElement()`解析数据库源。

### 获取SQL

SqlSessionFactoryBuilder.build (java.io.InputStream)

​	SqlSessionFactoryBuilder.java

​		XMLConfigBuilder.parse

​			XMLConfigBuilder.parseConfiguration

​				XMLConfigBuilder.mapperElement

​					XMLConfigBuilder.buildStatemnetFromContext

​						addMappedStatement

​							

mybatis加载mappers有4种方式:

resource, url, class, package

其中package优先级最高。

`mapperElement()`解析数据库源。

### 操作数据库

`openSession()`

mybatis有3种执行器：

simple, batch, reuse。默认是simple。

## 注解 vs XML

XML的优先级是比annotation高的。

正常interface无法实例化，但是在mybatis中的Interface加了注释以后会被执行。因为底层通过加代理进行了实例化。

Annotation: 不适合复杂sql。不方便管理sql

XML：条件不确定的查询。容易写错，特殊字符转义。

## SQL语句替换

MappedStatement.java中`getBoundSql()`方法。

SqlSourceBuilder.java中调用`parse()`方法进行了替换。