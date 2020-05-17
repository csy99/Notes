# 简介
创建一个类首先需要编译器将java文件进行编译，因为
jvm只认识class字节码。之后再用加载系统加载。

加载过程：
加载loading -> 验证verification -> 准备preparation -> 解析resolution -> 初始化initialization 

## 四种加载器
自定义类 - 系统类 - 扩展类 - 启动类

### 双亲委派模型parents delegation
优点：
1. 具有优先级层次的关系可以避免模板(类)的重复加载
2. 安全考虑可以防止Java核心api被替换

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

