# JVM体系架构
java是跨平台的语言，JVM是跨语言的平台（Kotlin,Scala, Jython, JRuby, JavaScript...)。JVM不关心运行在其内部的程序的编程语言，只关心字节码文件。

特点：

1. 一次编译，到处运行
2. 自动内存管理
3. 自动垃圾回收

运行在OS上，与硬件无直接交互。

## 不同种类的JVM

### Classic VM

第一款商用JVM。在JDK1.4被淘汰。只提供解释器(没有JIT)，执行效率低下。

### Exact VM

准确式内存管理Exact Memory Management。知道内存中某个位置的数据具体是什么类型。具备现代高性能VM的雏形。

### HotSpot VM

在JDK1.3时被设置成默认VM。热点代码探测技术。通过计数器找到最具编译价值代码，在本地缓存起来。通过编译器和解释器协同工作，平衡程序响应时间和最佳执行性能。

### JRockit

专注于服务器端应用，不关注程序启动速度，内部不包含解析器实现，是世界上最快JVM。MissionControl服务套件以最低的开销进行资源分析、管理和监控。

### J9

IBM。市场定位与HotSpot类似。多用途VM。通用性没有JRockit好。

## 基础

#### 代码执行流程

前端编译器(javac)将java文件进行编译成为class字节码。在JVM中，用类加载器进行加载。之后：解释器翻译字节码(解释执行)成机器语言，后端编译器(JIT)编译执行，将热点代码缓存起来。这两者协同工作。解释器响应快，但是JIT在正常运行之后(一开始有暂停时间)速度更快。

### JVM架构模型

**基于栈的指令集架构**

1. 设计和实现更简单，适用于资源受限的系统
2. 避开了寄存器的分配难题
3. 绝大部分指令是零地址指令
4. 可移植性好

**基于寄存器的指令集架构**

1. 典型应用：x86
2. 完全依赖硬件
3. 性能优秀，执行高效
4. 一/二/三地址指令

### 生命周期

**启动**

引导类加载器Bootstrap class loader创建一个初始类，这个类由虚拟机的具体实现指定。

**执行**

程序开始执行到结束。

**退出**

正常结束，程序出现异常或错误，操作系统出现错误，线程调用Runtime类或System类的exit方法。



# 类加载子系统

负责从文件系统或网络中加载class文件，有特定的文件标识。class文件的运行是有执行引擎Execution Engine决定。加载的类信息存放在方法区(还会存放运行时常量池信息)。

### 结构

<img src="./pics/01jvm_structure.jpg" width="500" height="300">

方法区和堆是多个线程共享的。

执行引擎只认机器码。

## 类的加载过程
加载loading -> 验证verification -> 准备preparation -> 解析resolution -> 初始化initialization 

### 加载

整体加载过程中的第一个环节。

通过一个类的全限定名获取类的二进制字节流。将其所代表的的静态存储结构转化为方法区的运行时数据结构。在内存中生成一个代表这个类的java.lang.Class对象，作为方法区这个类的数据的访问入口。

### 链接

#### 验证Verify

确保class文件中字节流中包含的信息符合VM要求，保证被加载类的正确性，不会危害VM的安全。

CAFE BABE。

文件格式验证、元数据验证、字节码验证、符号引用验证。

#### 准备Prepare

为类变量分配内存，设置该类变量(静态变量)默认初始值。这里不包含用final修饰的static，因为final在编译的时候就会分配了，准备阶段会显示初始化。

#### 解析Resolve

将常量池内的符号引用转换为直接引用的过程。

### 初始化

执行类构造器方法\<clinit\>()的过程。此方法不需定义，是javac编译器自动收集类中的静态变量的复制动作和静态代码块中的语句合并二来。

构造器方法中指令按照语句在源文件中出现的顺序执行。

若该类具有父类，JVM会保证父类的clinit先执行。 

## 两种加载器

四者之间的关系是包含关系，不是继承关系。扩展类和系统类算是自定义类。

获取加载器的方式。

```java
// 1获取当前类的加载器
clazz.getClassLoader();
// 2获取当前线程的加载器
Thread.currentThread().getContextClassLoader();
// 3获取系统的加载器
ClassLoader.getSystemClassLoader();
// 4获取调用者的加载器
DriverManager.getCallerClassLoader();
```

### 自定义类User-Defined

所有派生于抽象类ClassLoader的类加载器(不光是程序员自定义的)。默认使用系统类加载器进行加载。

使用场景：隔离加载类，修改类加载的方式，扩展加载源，防止源码泄露。

步骤：

1. 继承抽象类ClassLoader。
2. 加载逻辑写在findClass()方法。
3. 可以选择直接继承URLClassLoader。避免自己重写findClass()和获取字节码流的方法。

#### 扩展类Extension

继承了URLClassLoader。间接继承了ClassLoader。使用getParent会获取到null，因为bootstrap class loader不是java编写的。

#### 应用AppClassLoader/系统类System 

继承了URLClassLoader。间接继承了ClassLoader。使用getParent会获取到扩展类加载器。

### 启动/引导类Bootstrap

用C语言实现，嵌套在JVM内部。Java中的核心类库(e.g., String类)使用引导类加载器进行加载。没有父加载器。

## 双亲委派模式parents delegation

JVM对class文件采用按需加载。

### 工作原理

1. 如果一个类加载器收到了类加载请求，先把请求委托给父类的加载器去执行
2. 如果父类加载器还存在其父类加载器，依次向上委托
3. 如果父类加载器可以完成类加载任务，就成功返回；否则，子加载器再自己尝试加载

### 优点

1. 具有优先级层次的关系，可以避免类的重复加载
2. 安全考虑，可以防止Java核心api被替换

### 沙箱安全机制

对于java核心源代码的保护。自定义的同名类不会被引导类加载器加载(会先加载jdk自带的文件)。



初始化了，这个阶段主要是对类变量初始化，是执行类构造器的过程。Class只是对 Klass的封装。

# 布局

## 堆
### 新生代
Eden区： 诞生于此。当这个区快满的时候，会进行垃圾回收。仍然留下的对象将迁入Survivor中的分区。
Survivor区： 总共有15个分区。每一次gc会不断移动survivor分区中的对象到另一个分区。最后一个分区的对象（或者超大对象）会被放入老年代。

### 老年代
老年代快满的时候，会有full gc。

### 判断对象是否存活
1. 引用计数法
	缺点：两个对象互相引用，别的对象并不需要这两个对象，这两个对象不会被删除。
2. 可达性分析法

## 栈

## 程序计数器
为不同的任务划分出不同的时间片，我们在切换任务的时候，需要一个记录者，能够记录我们这个任务做到了哪里，下次回来能够继续做。

## 方法区


# 垃圾回收
## 标记-清除
缺点：1.效率不高  2. 会产生许多碎片空间
## 复制
缺点：缩小了一半空间。
## 标记-整理
首先标记出所有需要回收的对象，然后将存活的对象移动到空间的一端，然后清理掉边界以外的对象。


# Reference
- [JavaGuide公众号文章](https://mp.weixin.qq.com/s?__biz=Mzg2OTA0Njk0OA==&mid=2247486887&idx=1&sn=4598ab47a20bb1ecb60ffdb835e7255d&chksm=cea2426cf9d5cb7a0f7af304cb48a3b3cc61327d5f662a9250d5664030733b59ac0528cde4a5&scene=126&sessionid=1589496835&key=1afad850a7d33676730029e8fb9cfdc1ab9cf9433920be7b1065c9dbc4f0d4014ec5b93994cb6f3a26d726db2298fe302698756cfb5b4937389f70e587ce2970a35e5b82db0abd12c204c299deb39bad&ascene=1&uin=Mjg4MTIzMzIzNA%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=A5HIH3xzXlBBTW7EvWwCIRU%3D&pass_ticket=uSepJ84Ea0hIFlo7Ko6IXBxiegJjjEcSIU76iidJ19%2BcpL2q175YHVMvOE4INuob)

