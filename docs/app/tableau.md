# File Types
workbook (.twb): stores a visualization without source data

bookmark (.tbm): stores a connection to a worksheet in another Tableau
workbook

packaged workbook (.twbx): stores extracted data and visualizations for
viewing in Tableau

data extract (.hyper or .tde): stores Tableau data as a filtered and
aggregated extract

data source (.tds): stores the server address, password and other info
required to access a data source

packaged data source (.tdsx): a zip file that contains a .tds file

# Operations

1.  新增需要通过计算得到的一个属性

Analysis -&gt; Create Calculated Field -&gt;

将需要的列名拖入。

2.  填充通过计算得到的结果（结果不会被保存）

点击Marks区域-&gt; 输入公式 (e.g.: \[列名\]\*\[列名\])

3.  筛选

右击rows或者columns的标签选择add filter。可以自己写筛选公式。

OR

将Dimensions中属性拖入Filters区域。在Wildcard选项中可以选包含特定名称的数据。Condition选项中可进行数学比较的选项。Top选项中可以让数据只显示前n项。

OR

使用参数进行筛选：右击Measures中选项create-&gt;parameter-&gt;

4.  概要

worksheet -&gt; show summary

5.  改变统计方法

点击Marks区域的标签 -&gt; add table calculation -&gt; calculation type

6.  预测

Analytics -&gt; Model -&gt; 将forecast 拖入图中

右击rows或者columns的标签选择more -&gt; custom

7.  排序

右击rows或者columns的标签选择sort

合并之后排序：按住ctrl选中Dimensions中多项，右击选中create-&gt;combined
field

8.  分组

按住ctrl选中sheet中多行，点group member -&gt; edit alias

右击Dimensions中需要分组的属性，edit group -&gt; include other

查找组中的某个成员：右击Dimensions中需要分组的属性，edit group -&gt;
find

9.  集合

按住ctrl选中sheet中多行，点create set

对集合中子集合进行求和Analysis -&gt; totals -&gt; add all subtotals

合并多个集合：在Sets中选中需要合并的集合，右击create combined set -&gt;

10.  交叉表

analysis -&gt; totals -&gt; show row grand total

11.  美化图表外观

Marks区域Color、Size等选项

Format-&gt; shading -&gt; row banding -&gt; pane

12.  展示模式

F7或者Presentation mode图表

13.  高亮图表

show me -&gt; (top right) highlight table -&gt;

14.  提示文本

Marks -&gt; Tooltip -&gt;可以自己添加注释或者-&gt;insert -&gt; sheets

15.  画图

show me -&gt; 选择合适图表类型 -&gt;

右击rows或者columns的标签 -&gt; Dimension -&gt; 散点图

右击Dimensions中需要画直方图的属性 -&gt; create -&gt; bins

treemap: show me -&gt; treemap -&gt;

16.  找出聚集

Analytics (panel) -&gt; Cluster -&gt;

Marks -&gt; Clusters -&gt;

17.  注释

右击图像 -&gt; annotate -&gt; area

18.  美化图表元素

点击图像需要格式化的元素 -&gt; Marks -&gt;

右击图像需要格式化的元素 -&gt; edit axis

更换直方图柱子颜色: 右击相关图例 -&gt; edit color

更换顺序直方图柱子：拖动相关图例

19.  增加趋势线等

abline: Analytics tab in the navigation panel -&gt; 双击trend line

取消趋势线：右击trend line -&gt; 反向勾选show trend line

CI: Analytics tab in the navigation panel -&gt; 双击 avg with 95% CI

取消CI：双击CI -&gt; remove

reference line: 右击图像 -&gt; add reference line -&gt; distribution
-&gt; value -&gt; quantiles

20.  地图

将Dimensions中对应geographic data标签(有一个小地球图标)拖入Marks -&gt;
改变标签从detail成size

layers: Map -&gt; map layers -&gt; data layer -&gt;

distance: point arrow(类似一个播放的标志) -&gt; radial selection -&gt;
选择一个点并向四周拖拽 -&gt; zoom out 之后将没有这一功能

pan: point arrow -&gt; pan -&gt; 使用鼠标可以拖动地图以查看其它区域

Map -&gt; map options -&gt; 可以调整、限制map的操作

define custom regions: point arrow -&gt; rectangular selection -&gt;
选择相应城市 -&gt; group members -&gt;

21.  Dashboard

new dashboard -&gt; 向需要的图表拖入 -&gt;

对其中一个图表进行操作：点击 -&gt; arrow-&gt; floating(实现漂浮)

添加文本框：Objects -&gt; text

filter: 点击图表 -&gt; use as filer(漏斗状) -&gt; 点击需要的行、列-&gt;
仅展示需要的subset

highlight: Dashboard -&gt; action -&gt; add action -&gt; highlight

URL: 选中需要外链的图表 -&gt; set URL -&gt; 输入正确网址

# Functions
countd(): count distinct  
