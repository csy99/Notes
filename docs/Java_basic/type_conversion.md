# int to String
有三种方法，其中第二种第三种效率差不多，要比第一种更快。 
```java
a+""
String.valueOf(a)
Integer.toString(a)
```

# String to int 
```java
int a = Integer.parseInt(str);
```

# int to char
最正确的方式：
```java
char c = String.valueOf(5).charAt(0);
```
简便方法：
```java
char c  = (char) ('0' + 5);
```

# char to int 
最正确的方式：
```java
char c = ‘5’;
String str = String.valueOf(c);
int a  = Integer.parseInt(str);
```
简便方法：
```java
char charNum = '5';
int num = char - '0';
```

# double to long
```java
Long l = new Double(3.0).longValue();
```

# long to double
```java
long l = 2L;
Double d = l.doubleValue();
```

# 其他
如果long或者double出现溢出，需要使用BigInteger和BigDecimal。引入import java.math.BigInteger（BigDecimal）。

使用方法如下： 

BigInteger b = BigInteger.valueOf(a);  
或者BigInteger b = new BigInteger(String.valueOf(str));

相加是b.add(a);





