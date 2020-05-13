# 要点总结：
1.	GNU Compiler Collection也叫gcc，是最流行的C编译器之一。可以在很多操作系统中使用。在命令提示符中使用gcc cards.c -o cards进行编译。编译过后，一个cards.exe文件将出现（Windows系统）。在gcc之后跟一个-g表示让编译器记录编译代码的行号。
2.	C语言没有内置数据结构，必须自己创建。
3.	使用栈的时候不用担心清理其中的变量，因为这个过程是自动的。但是一旦申请堆上的空间，这块空间就不再被分配出去。存储器泄露是C程序中最常见的错误且难以追踪。
4.	valgrind通过拦截malloc()和free()检查存储器泄露问题。在valgrind.org上查看最新发行版的详细信息。
5.	指针是存储器中某条数据的地址。使用指针可以避免副本并且共享数据。使用“&变量名”找出变量的存储器地址。\*运算符与&运算符正好相反，它接收一个地址，返回地址中保存的数据。也被描述成对指针进行解引用。*运算符还可以设置存储器地址中的内容。
6.	对每种类型的数据，指针变量都有不同的类型。如果对char指针加1，指针会指向存储器中下一个地址，因为char只占1字节。如果对int指针加1，代码会对存储器地址加4，因为int占4字节。
7.	C语言不支持现成的字符串。但有很多扩展库提供字符串。字符串以数组的形式储存，定义的时候长度为n+1，因为字符串末尾需要有一个结束字符‘\0’。计算机会为字符串的每一个字符以及结束字符在栈上分配空间，并把首字符的地址和变量关联起来。
8.	一个数组变量就好比一个指针。所以在使用sizeof()运算符（不是一个函数，运算符在编译时就被分配了空间）的时候会返回奇怪的结果——只返回字符串指针的大小。我们定义一个数组d: d\[0\]与\*d等价。
9.	如果是普通的变量声明，char d\[ \] = “dump”; 就是一个数组。但如果是以函数参数的形式声明，那么d就是一个char指针，例如void test(char d\[ \])。
10.	指向字符串字面值的指针变量不能用来修改字符串内容，例如char \*d = “dump”; 不能用这个变量修改这个字符串。但如果用字符串字面值创建一个数组，例如char d\[ \] = “dump”; 就可以修改。如果我们想把指针设成字符串字面值，必须确保使用了const关键字。若是编译器发现有代码试图修改字符串，就会提示编译错误。
11.	但是数组变量与指针不完全相同。参考以下代码：
```c
char s\[ \] = “How big is it?”;
char \*t = s;
```
sizeof(s)会返回15，sizeof(t)会返回4或者8（取决于操作系统）。
数组变量不能指向其他地方。当创建指针变量时，计算机会为他分配存储空间。计算机会为数组分配存储空间，但是不会为**数组变量**分配任何空间，编译器会在出现它的地方把它替换成数组的起始地址。如此例中，s=t会报编译错误。
12.	指针退化：把数组赋值给指针变量，指针变量只会包含数组的地址信息，不包含数组长度。只要把数组传递给函数，数组免不了退化为指针，所以需要记清楚代码中发生过数组退化的地方，以避免引发不易察觉的错误。
13.	bus error（总线错误），意味着程序无法更新某一块存储器空间。
14.	布尔值用数字表示，数字0代表假，其他均代表真。
15.	使用break语句可以跳出循环语句和switch语句，但是不能跳出if语句。
16.	main()函数返回类型是int，不要忘记return。main()中调用函数必须在main()之前出现。有两种main()函数，一种没有传参，另一种有传参(int argc, char *argv[ ])。
17.	可以使用链式赋值如y=x=0。
18.	程序运行时，操作系统创建三条数据流：标准输入、标准输出和标准错误。但我们可以用fopen()函数创建自己的数据流。
19.	可以在数据类型前加关键词来改变数值的意义。用unsigned修饰的数值只能是非负数。用long修饰可以保存范围更广的数字。用extern修饰表示共享变量。
20.	函数声明指告诉编译器函数会返回什么类型的语句。这是为了防止编译器假设函数的返回类型。
21.	创建头文件的步骤：创建一个扩展名为.h的文件，然后在主代码中包含头文件。用引号把文件名括起来，让编译器在本地通过相对路径查找文件。
示例：#include “statement.h”
22.	用结构创建结构化数据类型（把一批数据打包成一样东西）。struct是structured data type的缩写。可以按名访问结构字段（f1.name）。当把一个结构变量赋值给另一个时，只会复制结构的内容。
示例：
```c
struct fish {
	const char *name; // 保存不想修改的字符串
	const char *species;
	int age;
}
struct fish f1 = {“fish1”, “big”, 4};
```
23.	可以用typedef为结构命名。创建别名后可以不需要结构名。但是在递归结构中，需要包含一个相同类型的指针，因此必须为结构起一个名字。
示例：
```c
typedef struct fish {  
    const char *name; // 保存不想修改的字符串
	const char *species;
    int age;
} ff; // ff将成为struct fish的别名
ff f1 = {“fish1”, “big”, 4};
```
24.	“指针->字段”相当于“(*指针).字段”。
25.	C语言不支持二进制字面值，但是支持十六进制字面值。

# 命令行操作：
1.	可以用<重定向标准输入，用>重定向标准输出，用2>重定向标准错误。
示例：
```program < data.csv > output.json```
2.	Windows的命令提示符中输入echo %ERRORLEVEL% 可以在重定向输出的同时显示错误消息。
3.	两个独立的程序用管道连接以后就可以看成一个程序。
示例：
```(program1 | program2) < data.csv > output.json```
4.	make编译的文件叫目标。对于每个目标，需要知道它的依赖项和生成方法（合在一起构成一条规则）。生成方法必须以tab开头。
示例：//将make规则保存在当前目录下一个叫Makefile的文本文件中
```shell
launch.o: launch.c launch.h thruster.h //依赖文件
　　gcc -c launch.c //生成方法
thruster.o: thruster.h thruster.c
　　gcc -c thruster.c
launch: launch.o thruster.o
　　gcc launch.o thruster.o -o launch
```
//在控制台输入以下指令
```shell
> make launch
gcc -c launch.c
gcc -c thruster.c
gcc launch.o thruster.o -o launch
```
5.	在多个C项目中共享头文件的方法：把头文件保存在标准目录中（/usr/local/include）；在include语句中使用完整路径名；告诉编译器去哪里找头文件（gcc后跟-I选项）。
6.	存档命令（ar）会在存档文件中保存一批目标文件。
示例：
```> ar -rcs libsecurity.a f1.o f2.o  ```
r表示如果.a文件存在就更新它，c表示不反馈信息，s告诉哦ar要在.a文件开头建立索引。后面是保存在存档中的文件。
所有.a文件名都是libXXX.a的形式。这是命名静态库的标准方式。
7.	当存档安装在标准目录，可以使用-l编译代码。
示例：```gcc  test.c -lsecurity -o test```
//security叫编译器去找libsecurity.a的存档
//可以设置多个-l选项来使用多个存档
当存档安装在其他目录，可以用-L编译代码。
示例：```gcc  test.c -L/myLib -lsecurity -o test```
//存档放在了/myLib  
8.	动态库在Windows中叫动态链接库(.dll)，在Linux和Unix上交共享目标文件(.so)，在Mac上叫动态库(.dylib)。一旦用某个名字编译了库，就不能修改文件名。重命名必须用新的名字重新编译一次。
创建方式示例：```gcc -shared test.o -o C:\libs\test.dll```

# 标准库：
在程序开头使用#include调用C标准库。标准库分了好几个部分，每个部分独有一个头文件，列出了该部分的所有函数。示例：#include <stdio.h>.
## <stdio.h>
1.	使用puts()或者printf()函数打印。%i格式化整型，%s格式化字符串，%p格式化地址。
2.	使用fprintf()打印到数据流。
示例：fprintf(stderr, “message”);
3.	使用scanf(“%ns”, var)从键盘读取n个字符（自动补全结束字符），存入var这个变量中。如果忘记限制scanf()读取字符串的长度，用户可以输入远超出程序空间的数据。多余的数据会写到尚未被分配好的存储器中。运气好的话，数据会被保存。但缓冲区溢出很可能导致程序出错，通常是segmentation fault（段错误）或者abort trap。不管出现什么错误消息，程序都会崩溃。
scanf()允许输入多个字段，允许输入结构化数据，可以指定两个字段之间以什么字符分割。但是遇到空格就会停止。
4.	fgets()与scanf()函数类似，接受char指针，但是必须有最大长度这个参数。语法为fgets(var, sizeof(var), stdin); 最后一个参数表示数据来自键盘。
fgets()只允许想缓冲区输入一个字符串，不能是其他数据类型。但是fgets()不受空格的限制。
5.	fopen()函数接受两个参数：文件名和模式。共有三种模式：w(写文件)，r(读文件)，a(在文件末尾追加数据)。
示例：
```
FILE *in_file = fopen(“input.txt, “r”);
```
6.	fclose()函数关闭数据流。虽然所有的数据流在程序结束后会自动关闭，但是我们仍然应该自己关闭它们。

## <string.h>
1.	strstr(a, b)函数会返回b字符串在a字符串中的位置。
2.	strcpy()可以复制字符串。
3.	strchr()用来在字符串中找某个字符的位置。
4.	strcmp()比较字符串。
5.	strlen()返回字符串的长度。
6.	strcat()连接字符串。
7.	strdup()计算出字符串的长度，然后调用malloc()在堆上分配相应空间，然后把所有自负复制到堆上的新空间。请记得使用free()释放空间。

## <stdlib.h>
1.	malloc()会要求操作系统在heap中分配空间，并返回一个指针指向这个空间。该函数接收一个参数：所需要的字节数。在不知道确切字节数的情况下，常与sizeof一起使用。
示例：```Person tmp = malloc(sizeof(Person));```
2.	free()函数可以释放存储器。
示例：```free(tmp);```
3.	qsort()是一个排序函数，判断两条数据的大小关系。
```qsort(void *array, size_t length, size_t item_size, int (*compar) (const void *, const void *)); //void*指针可以指向任何数据。``` 

在写comparator的时候，第一件事是从指针中提取值，因为值以指针的形式传给函数。
示例：第一个\*拿到保存在地址中的值，第二个把void指针转换为整型指针
```int a = *(int*) score_a;```

## <stdarg.h>
所有处理可变参数函数的代码都在这个库中。可变参数将保存在va_list中。可以用va_start()、va_arg()和va_end()控制va_list。可变参数函数至少需要一个普通参数。读取参数时不能超过给出的参数个数，且需要知道参数类型。


# 非标准库：
不属于C标准库。
## <unistd.h>POSIX库
1.	getopt(): 这个库函数每一次调用都会返回命令行中下一个参数。
程序需要两个选项，一个选项接受值，-e代表“引擎”；另一个选项代表开或关，-a代表“无敌模式”。
示例：
//最后一个参数表示选项a和e有效冒号表示e需要一个参数
//getopt()会用optarg变量指向这个参数
```getopt(argc, argv, “ae:”)  ```
读完全部选项以后，应该用optind变量跳过它们。
示例：
```
argc -= optind;
argv += optind;
```


