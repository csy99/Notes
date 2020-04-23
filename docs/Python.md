# 变量类型及基本操作
## 赋值方式
### 单个赋值
```
x=1
y=2
```
### 多个赋值
```x, y = 1,2```
### 用列表赋值
```x, y = [1, 2]```
### 用字符串赋值
字符个数需要与对象个数相等，且赋值后对象类型为字符串
```x, y = ‘12’```

## 打印
调用print()方法时，相当于调用了java中的println()。  
括号内用逗号隔开不同元素，此时不需要将各个元素转换成字符串，且元素之间自动生成空格。  
用+连接不同元素需要将各个元素转换成字符串，且元素之间不会自动生成空格。  
print()中调用end参数决定下一次print的开始位置。默认是end=’\n’，也就是另起一行。  
```print(‘str %s %d str%.2f’ %(x,y,z))```String用%s，int用%d, float用%.2f（表示保留两位小数）。  

## 警告
消除所有警告
<pre name="code" class="python">
import warnings
warnings.filterwarnings(‘ignore’)
</pre>

## 列表
i.	可用[]定义空列表。  
ii.	运用extend方法可以将列表或者元组中的元素拆分后单独加入列表中。  
iii.运用append方法可以将单个元素，整个列表或元组以整体形势计入列表中。  
iv.	+=操作与extend方法类似  
v.	运用\[start at : end before]将列表切片。  
vi.	\[:-num]返回将最后n个元素去除后的列表  
vii. sorted(list)返回有序列表但并不改变列表本身  
viii. 列表调用.sort()方法将列表排序且改变列表本身，但不返回值  

## 元组
i.	可用()定义空列表。  
ii.	与列表的区别：元组形成后不可更改，只可以用+=操作增加。设计理念的不同使得元组调用起来更省时间  
iii.有切片功能  
iv.	可使用sorted(tuple)，但是元组没有sort()  
v.	元组内只有一个元素的时候在后面加逗号，如(1,)  

## 字典
i.	可用{}定义空列表。  
ii.	键与键值相配对。调用时使用键而非指数作为索引  
iii.给一个不存在的键赋值会直接将这个键加入字典中  
iv.	删除：```del dict[‘键’]```将该组键与键值一起删掉  
v.	更改：
```
dict[‘新键’] = dict.pop[‘老键’] 
dict.update({‘新键’: dict.pop(‘老键’)})
# 先删后加
```
vi.	groupby()

## 集合
i.	可用set()定义空列表。  
ii.	唯一性：同一元素最多存在一个  
iii. 不支持索引  
iv.	set(字符串)会将其转成单个字符的值进行插入  
v.	add()用于单个元素的增加   
vi.	update(\[])用于多个元素的增加  
vii.	remove()用于单个元素的删减，且元素必须存在于集合中  
viii.	discard()用于单个元素的删减，但元素可以不存在于集合中  
ix.	pop()随机返回一个元素值，并删除。集合为空时报错  
x.	clear()清空整个集合  
xi.	set1.union(set2)返回一个包含两个集合中所有元素的新集合   
xii.	set1.intersection(set2)返回一个包含两个集合中相同元素的新集合   
xiii.	set1.difference(set2)返回一个set1有但set2中没有的元素的新集合  
xiv.	set1.symmetric_difference(set2)返回两个集合中不同元素的新集合   

## 其他
zip()可以把多个列表或者数列中每个元素对应位置的书集合在一起。首先这三个列表要一样长，把每个位置相同索引的数放到一起。
e.g.:  
```
list1 = [1,2,3]
list2 = [4,5,6]
list3 = [7,8,9]
list = list(zip(list1,list2,list3))
```
里面会存有[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

## 日期
```WeekdayLocator(Monday)```表示定位到特定的一天  
```DayLocator()```表示以一天为最小的单位来表示  
```DateFormatter(‘%m %d, %Y)``` # Oct 12, 2018  
```date2num(list)```表示将列表转换成日期形式  
```date.isocalendar()```返回一个包含三个参数的元组(年，周，星期几)。最后一个参数值域为1-7,1表示周一  

# 文件读写
## 普通读写
fl = open(‘路径’,'操作')
第二个参数如果是r，则读取文件；如果是w，则写入文件。  
读取文件后，采用.read()会按原文件格式行成字符串，采用.readlines()会将原文件按行存进一个列表。  
写入文件调用.write()。  
## 调用pandas
```df = pd.read_csv(‘路径’, sep = ‘,’, index_col = 0)  ```
基本上一种文件对应一个pd的read方法。读取xlsx文件调用.read_excel(), 读取csv文件调用.read_csv(), 读取txt调用.read_table()。  
sep表示分隔符的标记，可省略。  
Index_col表示以读取内容的哪一列作为表格第一列，可省略。如果省略，pd会自动生成一列序数列。   
```df.head(# of lines)```显示数据前n行，n默认值为5。df.tail()显示最后n行。  

#将DataFrame存储为csv, index表示是否显示行名(默认为真)
```df.to_csv("Predict.csv", index=False, sep=',')```
## 逻辑
退格缩进很重要，Python不像java有大括号框定范围。记得加冒号。
### 条件判断
```
x = {1, 2, 3}
if 3 in x:
	…
elif 4 in x: 
	…
else:
	…
```
注意Python中不能用java（!Bool）的方式表示Bool的否定来进行判断。
### 循环
i.	for  
```
for i in [1,3,6,9]: #列表中的每一个值，也可以循环字符串  
for i in range(4): #循环0-3  
for i in range(0, 30, 2): #循环1-29，步长为2
for i in df.index: #循环数据框的每一行
```
ii.	while
语法与if类似
### pass, continue, break
pass是占位符，让代码更加完整。因为定义的函数，条件判断或者循环中的内容为空的话，程序会报错。  
continue继续下一个循环。  
break跳出当前循环。  

# 函数
def 函数名(参数):
	return var

# Pandas
以numpy数据结构为基础构建的库。
## 序列
```
s1 = pd.Series([1,2,3,4,5]) # 转换成序列
s1.values # 返回所包含的值
s1. index # 返回起始点和步长。非等差数列终止点和步长返回一个不太靠谱的数值。
```
索引方式与列表、元组相同

```s2 = pd.Series([1,2,3,4,5], index = [‘a’, ’b’, ’c’, ’d’, ’e’])```索引方式与字典相同  
```s1.index = [‘a’, ’b’, ’c’, ’d’, ’e’]``` 重新给索引赋值  
```s1 + s2```如果索引相同，序列可以直接相加；否则返回NaN。  

```s1.isnull()``` 未被赋值过的索引返回True  
```s1.notnull()``` 反之亦然   

## DataFrame
### 打印
```
dat1 = {‘A’:[1,2,3], ‘B’:[3,4,5], ‘C’:[5,7,9]}
df1 = pd.DataFrame(dat1)
print(df1) # 将这组数据以表格形式打印出来
```

```
dat2 = {‘A’:1, ‘B’:2, ‘C’:3}
df2 = pd.DataFrame(dat2)
print(df2) # 当数据只有一个值的时候会报错，因为不是列表形式
```
需要修改第二行代码：
```df2 = pd.DataFrame(dat2, index = [0])``` 
index是每一行的索引  

### 指定列名
字典法
```
d  = {‘one’ : pd.Series([1,2,3], index = [‘a’,’b’,’c’]), ‘two : pd.Series([1,2,3,4], index = [‘a’,’b’,’c’,’d’])}
df3 = pd.DataFrame(d) # 行名是index，列名是字典名
df.index # return Index(index = [‘a’,’b’,’c’,’d’], dtype=’object’)
df.columns  # return Index([‘one’, ‘two’], dtype=’object’)
```
### 索引
1. 默认按列进行索引，直接按行进行索引会报错。填入一列进行索引时，类型为序列；若是填入多列进行索引，类型是DataFrame。
想要返回特定数据，使用df[column][index]进行索引。index可以采用切片，且后一个数值表示end at而不是像列表一样的end before。当需要选择多列的时候，用列表表示column变量，元素用逗号隔开。不可以使用“:”操作。
2. 调用loc和iloc函数。
loc函数使用指定的行/列名
df.loc[[行][列]] #用逗号将所需要行/列隔开  
df.loc[行start at : end before,列start at : end before]  
iloc函数使用默认的行/列名（从0开始，依次递增）  
语法与loc函数相同  
3. 利用条件进行切片
e.g.  
```df[df[‘A’]>3]``` 选取A列值大于3的行  
```df[df<10]``` 打印整个数据结构，显示小于10的值，其他用NaN表示  
```df[(df>3)&(df<10)]``` # 用”&”或者”|’进行逻辑连接  
4. 对不存在的列进行赋值，会直接增加这一列（与字典类似）。
5. 调用apply()函数，默认对每一列进行操作。可以调整参数axis = 1，使得对每一行进行操作。如果是利用切片功能获取了一列再调用apply()，则是对此列中每一个数进行操作。
e.g. ```df.apply(np.sqrt)```
在apply()使用lambda参数进行简单的操作。
e.g. ```df.apply(lambda x: x+1) ```仍然是以列为单位进行操作
	- 对其中一列调用.pct_change()函数可以计算百分比差值。
	- 对其中一列调用.shift(m)函数可以返回该单元格在此列向上m行的值，m可以是负数。
### 删除元素
```
df.drop([‘row index’])
df.drop(‘column name’, axis = 1) # 删除列
```


## 格式
### DateTime
将包含时间的字符串格式的Date变量转换成时间戳格式  
```df1['Date'] = pd.to_datetime(df1['Date'] , format = '%Y-%m-%d')```
### Numeric
将其他格式转换成数字格式```pd.to_numeric```

## 合并
### Concatenate
```pd.concat([df1, df2, df3])```
其他参数：
- axis = 0 (默认)，以列为标准对齐(合并相同名字的列，显示所有行，行名重复不会报错。缺失数据以NaN表示。)；axis = 1， 以行为标准对齐。
- dropna(how = ‘any’)只要该行有NaN则该行不会显示；dropna(how = ‘all’)当该行所有值均为NaN时该行不会显示。
- join = ‘outer’（默认），返回全部数据；join = ‘inner’，返回共有数据；join_axes = [df1.index]保留df1中所有数据，尝试合并其他表中数据。
- ignore_index = True，忽略行/列名，以序数替代。
### Merge
```pd.merge(left, right, on = ‘公共列’) ``` 最多合并两个表
其他参数：
- on = ‘公共列’，合并后公共列只显示一次。
- how = ‘inner’ (默认)；how = ‘outer’；how = ‘left/right’。意义与concat函数的join参数相同。
- left_index = True, right_on = 右表名[‘右表列’] (这种形式解决了两个表中有相同列名但正好不是要被合并的列的问题。如果不存在这个问题，可以使用right_on = ‘右表列’）。这组命令可以配合使用，表示使用左表的index与右表的列进行匹配输出结果。也可以配合使用right_index和left_on参数。
### Join
```left.join(right) ```相当于以左表为基础做merge
其他参数：
- how = ‘inner’；how = ‘outer’。意义与concat函数的join参数相同。

## 排序
-	align()
默认outer join.
-	rank()
-	sort()
df.sort_index(axis = 1, ascending = False)
df.sort_value(axis = 1, ascending = False)


## 补充缺失值
df.fillna(value  = 10)
## 混合
pd.melt()


# Numpy
## 随机数
```np.random.randn(3) ``` 返回3个随机数
```np.random.randn(10, 4)``` 返回10*4(矩阵形式)个随机数
注意： 
numpy.random.randn()是从标准正态分布中返回一个或多个样本值。  
numpy.random.rand()的随机样本位于\[0,1)中。  
## 多维数组变换
```
X = np.array([ [1, 2, 3 ], [1, 2, 3 ] ])
X = X.flatten()
```
转换为一维数组
## 生成数据
```
x = np.arange(0, 10, 0.1) #以0.1位单位，生成0-10的数字
x.reshape(3,5) #转换成矩阵
np.zeros((3,4))
np.empty((3,4))
def func(x,y):
	return 10*x + y
np.fromfunction(func, (4,5))
```
## 正余弦
```np.sin(x)```
## 输入输出
```
np.loadtxt(‘file.txt’)
np.genfromtxt(‘file.csv’, delimiter = ‘,’)
np.savetxt(‘file.txt’, arr, delimiter = ‘ ’)
np.savetxt(‘file.csv’, arr, delimiter = ‘,’)
```

# matplotlib
## matplotlib.pyplot
折线图： 
```
plt.figure(figsize=(10,5)) #定义图片大小
plt.plot(x,y1,label = ‘A’) #代入数值并画出图片（未显示在屏幕上），默认为折线图
plt.plot(x,y2,label = ‘B’) #多组数据进行对比
plt.title(‘’) #图片标题
plt.xlabel(‘’) 或plt.ylabel(‘’) #轴标题
plt.legend((‘A’,’B’), fontsize = 10, loc = 1) #给两组数据做标记并设置字体大小和显示位置（如果画图时没有使用label参数进行标记，此处标记要和上面画图的顺序相对应；如果画图时已经进行了标记，此处可以省略使用元组进行标记）
plt.show() #将图片打印出来
```

柱状图： 
```plt.bar(x,y, color = ‘#9999ff’, width = 0.5) ``` 设置颜色和柱子的宽度。  
柱状图想要进行两组数据的对比需要改变两组数据x的坐标(柱子的中心点)，否则柱子会重叠在一起。  
例：
```
x1= [0.25, 1.25, 2.25, …]
x2= [0.75, 1.75, 2.75, …]
plt.bar(x1, y1, width = 0.5) 
plt.bar(x2, y2, width = 0.5) 
```

点状图： 
```
plt.scatter(x,y)或者plt.plot(x, y, ’.’)
# 加趋势线
plt.plot(x,x)
plt.legend((‘real data’, ‘fitted line’))
```

盒状图： 
```
plt.boxplot(x) # 研究outlier
plt.boxplot((x,y)) # 也可以把两组数据画在一个图上
plt.xticks([1,2], [‘A’, ‘B’])或plt.yticks() # 将轴的标记从原来的值([1,2])更改成新的值[‘A’, ‘B’]
```

直方图： 
```plt.hist(x, # of pillar, color = ‘navy’, alpha = 0.5) # alpha调色彩饱和度```

其他技巧：
改变x轴显示值且并不标出所有值（易读）：
```
k = list(range(0,100,10)) #设定间隔长度
plt.xticks(df1.index[k], df1.Date[k])
```
ticks中r'$\pi/2$' 显示π/2 (Latex)

简便画图：
```
df1.index = df1['Date'] #注意Date不能是字符串格式
df1['Close'].plot()
```

同时显示多个图：
```
fig = plt.figure(figsize = (20,40)) #包括了所有subplot范围的区域
ax = fig.add_subplot(4,2,1) #一共四张小图，每一行显示两张，这个是第一张图
df1.High.plot(label = 'High') #High均为列名
```

对图进行注释：
```
fig = plt.figure(figsize = (20,40)) 
ax = fig.add_subplot(1,1,1)
x = df1.Close[:80]
y = df2.Close[:80]
plt.scatter(x,y)
ax.annotate(str(df1.index[40]), (x[40], y[40])) #左边参数为标记内容，右边参数为被标记对象
```

## 显示图像
```
from matplotlib.image import imread
img = imread(‘path’)
plt.imshow(img)
plt.show()
```

# Seaborn
```
import Seaborn as sns
sns.regplot(df1, df2) #自动添加回归线和CI
#画出分布图
sns.distplot(df_train['SalePrice']);
#画散点图加回归线
sns.lmplot(x=’attack’, y=’defense’, data=df)
sns.lmplot(x=df.attack, y=df.defense) #另一种写法
# 不想要reg line的话设置fit_reg = False
#画小提琴图
sns.set_style(‘whitegrid’)
sns.violinplot(x = , y=, data = )
#画蜂群图
sns.swarmplot(x=, y=, color=, alpha=0.7) #alpha设置透明度
#热图
sns.heatmap(df.corr())
#统计图
sns.countplot(x = ,y =, data=) #和barplot很像
#因子图
g = sns.factorplot(x=, y=, hue=, col=, kind=’swarm’)
g. set_xticklabels(rotation = 30) #x坐标旋转
```
# Urllib
```import urllib.request```  
读取网站数据：
```
file = urllib.request.urlopen(url)
data = file.read()
js = data.decode(‘utf8’) #用utf8格式解码
file.close()
parse_data = json.loads(js) #转换成json文件
ps = parse_data[‘表名’]
df = pd.DataFrame.from_dict(ps, orient = ‘index’) #用index做每一行的名字，默认是每一行的键名
```


# Scipy
```
from scipy.optimize import minimize 
def f(x):
	return …
#x0是初始值， 可以是一个np.array
res = minimize(f, x0, method = ‘nelder-mead’, option={‘xtol’:1e-8, ‘disp’: True， ‘maxiter’: 1000 })
```


```
import scipy.stats as st
y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

y = train['SalePrice'].values
def johnson(y):
    gamma, eta, epsilon, lbda = stats.johnsonsu.fit(y)
    yt = gamma + eta*np.arcsinh((y-epsilon)/lbda)
    return yt, gamma, eta, epsilon, lbda

def johnson_inverse(y, gamma, eta, epsilon, lbda):
    return lbda*np.sinh((y-gamma)/eta) + epsilon

yt, g, et, ep, l = johnson(y)
yt2 = johnson_inverse(yt, g, et, ep, l)
plt.figure(1)
sns.distplot(yt)
plt.figure(2)
sns.distplot(yt2)
```

# mpl_finance
```
from mpl_finance import candlestick_ohlc  
candlestick_ohlc(ax, quotes, colorup='red', colordown='green', width=0.5)
# 前两个参数必填。ax指的是Axes，即这一个小图的位置。Quotes填数据的必要信息。这里需要携程sequence of sequencies，使用date2num函数。

data['7d'] = data['Close'].rolling(window=7, center=False).mean()  # 移动平均数，window是选取几天的平均数
```