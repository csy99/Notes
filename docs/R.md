# 排序
对数据集进行排序的时候，不打乱行之间的联系
#sort m by using the order and keep the connections. 
#Don't miss the comma after the order. 
sort_df <- m\[order(m$attr), ] 

# 画图
##	自定义坐标
在plot()函数中可以直接使用at属性
在hist()函数中需要：
<pre name="code" class="R">
hist(变量, xlim = range(0,10), xaxt="n") #将x轴取消
axis <- axis(1, at=seq(from, to, step)) #1代表横轴
</pre>
##	将图片直接画到需要导出的pdf中
<pre name="code" class="R">
pdf("test.pdf")
plot()
dev.off()
</pre>
##	对数据集中所有数据画相关性散点图
```pairs(m)```  
如果想通过数字形式呈现  
```cor(m[ ,i:j])```

##	使用barplot()画图报错height必须为向量
<pre name="code" class="R">
#将数据转换成table形式
barplot(table(data))
</pre>

#	线性回归
## 预测
<pre name="code" class="R">
d = data.frame(var_name = c() )
predict(m, newdata = d)
</pre>
## y=a+bx最小二乘直线
<pre name="code" class="R">
abline(a = y.intercept, b= slope)
</pre>
或者
<pre name="code" class="R">
m = lm(y~x)
abline(reg = m)
</pre>
## 检查误差（误差均值应该为0）
<pre name="code" class="R">
plot(m$fitted.values, m$residuals) 
</pre>
或者
<pre name="code" class="R">
qqnorm(m$residuals)
</pre>
或者
<pre name="code" class="R">
layout(matrix(data=1:4, nrow=2, ncol=2, byrow=T))
plot(m)
</pre>

## 检查数据是否为常态分布
<pre name="code" class="R">
qqnorm(m); qqline(m)
</pre>
