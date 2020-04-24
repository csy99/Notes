# Data Type and Structure  
## template
类似于Java的泛型
e.g.:
```
template <typename T>
T maxof (const T &a, const T &b){
	return (a > b ? a : b);
}
```
使用maxof<int> (a, b)调用。// a和b均为int类型  
在link阶段，如果出现build failure，就把.cpp文件中的template的实现移动到.h文件中。
编译器会出现argument deduction的推断，有时候可以不指名template所需要的变量类型（红色标注）。
## auto
系统自动适应变量类型，程序员不需要花费时间去了解一个变量声明时的类型。
## Include Guard
a technique which uses a unique identifier that you #define at the top of the file
e.g.:
```
//x.h
#ifndef __X_H_INCLUDED__   // if x.h hasn't been included yet...
#define __X_H_INCLUDED__  
class X { };
#endif
```
the right way to include  
- **\#include "b.h"** if: B is a parent class of A; or, A contains a B object: B myb;
- **forward declare B** if: A contains a B pointer or reference: B* myb; or, at least one function has a B object/pointer/reference as a parementer, or as a return type: B MyFunction(B myb);

## vector
Constructor:   
```vector<int> v4(v3);  // 把v3内容全部复制过来```  
```vector<int> v5(std::move(v4));  // 把v4内容全部剪切过来```  

## list
Initiate：
```list<int> l1 = {1,2,3,4}; ```  
Modify:
.remove(T)直接清除list中的元素，而非位置  
清除两个Iterator之间的元素
```
auto it2 = it1 = l1.begin();
while( (*++it1 != 100) && (it1 != l1.end()) );
while( (*++it2 != 200) && (it2 != l1.end()) );
if(it1 != l1.end() && it2 != l1.end()){
	l1.erase(it1, it2);
}
```

## Pair & Tuple
```pair<int, string> p1(1, “one”);```  
如果是大于三个元素，需要自己写print方法  

## Array
Initiate：
```array<int, 5> a1 = {1,2,3,4,5};```  
Get:    
	a1\[1\]相当于a1.at(1)  
	.data()返回一个iterator（已声明数据类型的指针）  
	
## Deque & Queue & Stack
// deque和list非常像  
  
## Set
提供hash key。multiset允许重复元素的出现。默认是根据alphabetical顺序排列，但是也有unordered_set。  
删除元素：
```
auto it = set.find(“ten”);
if(it != set.end()){
	set.erase(it);
} else {
	cout << ”not found\n”;
}
```
    	
## Maps
.insert({ })方法括号中一定是一个大括号。该方法有三个返回值，第一个是插入元素(或阻止插入的元素)的迭代器，第二个是布林值指示插入操作是否成功。分别用.first和.second来取。
.find()方法返回指针。  
multimap()可以有重复键，insert()方法不再检查，所以不会返回布林结果。   

## Iterators
遍历
```
for(auto it = v.begin(); v < v.end(); ++it) { }
for(auto it = set.end(); it != set.begin();){
	cout << *--it1 << “ ”;
}
auto it2 = v1.begin() + 5;  // *it2是第6个元素
auto it3 = v1.end() - 5;  // *it3是倒数第5个元素
```  
读取
```
cin.clear();
istream_iterator<double> eos;  // default constructor is end-of-stream
istream_iterator<double> iit(cin);
if(iit == eos))
	cout << “no values” << endl;
else
	d1 = *iit++;
```
输出
```
ostream_iterator<int> output(cout, “ ”);
for(int i : {1, 10, 100}) {
	*output++ = i;
} 
```    

## Transform
```
template<typename T>
class accum{
	T _acc = 0;
	accum() {}
public:
	accum(T n)  :  _acc(n){}
	T operator() (T y) {  _acc += y; return _acc; }
}
accum<int> x(7); 
cout << x(7) << endl;  // 此时_acc=14
vector<int> v1 = {1,2,3,4,5};
vector<int> v2(v1.size());
transform(v1.begin(), v1.end(), v2.begin(), x);
transform(v1.begin(), v1.end(), v2.begin(), [accum](int d) mutable -> int {return accum += d; });  // 效果与上同
```

```
template<typename T>
class embiggen {
	T _accum = 1;
public:
	T operator() (const T & n1, const T & n2) { return _accum = n1 * n2 * _accum;}
};
vector<long> v1 = {1,2,3,4,5}
vector<long> v2 = {5,10,15,20,25}
vector<long> v3 = {v1.size(), 0}
embiggen<long> fbig;
transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), fbig);  // 结果存放在第4个arg
```

## Functors
\#include \<functional>
可以结合transform，sort使用
### Arithmetic Functors
plus/minus/multiply/divide/modulo/negate<T>
### Relational Functors
greater/less/greater_equal/less_equal/equal_to/not_equal_to<T>  
e.g.: 
```
greater<long> f;
sort(v2.begin(), v2.end(), f);  // 从大到小排序
```
### Logical Functors
logical_and/logical_or/logical_not<T>
         
         
# STL Algorithm
\#include \<algorithm>
## 搜索 
all_of/any_of/none_of(v.begin(), v.end(), function<T>)   
find()  // return an iterator point to the element if found, otherwise point to end  
find_if/find_if_not/search/count/count_if()  
binary_search(v.begin(), v.end(), n)   记得先sort(v.begin(), v.end());  
## 修改  
replace(v.begin(), v.end(), 47, 99)  
replace_if/remove/remove_if/unique  
## 复制  
copy(v1.begin(), v1.end(), v2.begin())   
copy_n()  // 指定copy的个数  
copy_backward(v1.begin(), v1.end(), v2.end())  // copy从后向前进行，不是顺序  
reverse_copy()  // copy the elements in reverse order   
reverse()  // reverse elements in place   
fill/fill_n()  // in place 
generate(v2.begin(), v2.end(), \[\]()->int { return rand() % 100;})  
random_shuffle(v1.begin(), v1.end())  // shuffle elements in place   
shuffle(v1.begin(), v1.end(), g)  // require one more arg: random function  
## 划分
partition(v1.begin(), v1.end(), f)  // partition in place using predicate function f  
stable_partition()  // 保留原来顺序  
partition_copy(v1.begin(), v1.end(), v2.begin(), v3.begin(), f)  // partition and store in different places  
## 排序
sort/stable_sort()  
## 合并
merge()  
