# 注意事项
1.	空格不计
2.	以分号结尾
3.	不区分大小写
4.	使用\<?php phpinfo(); ?>检测文件
5.	在php.ini文件中打开错误提示：
```
display_errors = On  
error_reporting = E_ALL
```
或者在php代码中写入
```
Ini_set(‘display_errors’, ‘On’);
Error_reprting(E_ALL);
```

# 语法
1.	\<?php php语句 ?>  
2.	\<br />换行  
3.	echo ”Hello world”; 在页面端显示，相当于打印语句  
4.	”Hello” . “ world”相当于”Hello world”, 句点可连接字符串  
5.	单行注释可以使用//或者#, 多行注释使用/* */  
6.	变量名以$打头，可以使用字母、下划线、数字、破折号，区分大小写  
7.	字符串中引入变量可采用“{$var}Hello”的形式  
8.	字符串允许使用.=进行赋值，相当于java的+=  
9.	字符串方法  
转换大小写：strtolower(), strtoupper(), ucfirst(), ucwords()等。最后一个方法会让句子中所有单词首字母都大写。在括号内放入需要进行转换的字符串  
其他操作：strlen(), trim(),   
strstr(strToBeOperated, toBeFound), 截取包含当前单词内的子字符串。  
str_replace(toBeReplaced, replaceStr, strToBeOperated)   
str_repeat(strToBeOperated, times)  
substr(strToBeOperated, start, lens)  
strops(strToBeOperated, “str”)， 寻找字符串，返回字符串首字母出现的位置  
strchr(strToBeOperated, “char”)， 寻找字母, 返回包含该字母开始后的子字符串  
10.	整数方法  
abs(), pow(), sqrt(), fmod(), rand(), rand(min, max) 上下界均包含  
is_int(), 正确返回1，错误不返回  
is_numeric()  
打印1 + “1”会出现2  
11.	小数方法  
round(floatToBeOperated, 1), 四舍五入到小数点后面几位  
ceil(floatToBeOperated), floor(floatToBeOperated)  
is_float()，is_numeric()  
12.	数组方法    
$arr = array(1, “fox”, array(“x”, “y”));或者$arr = [1, “fox”, array(“x”, “y”)];  
不能使用echo $arr\[2]将数组中的数组直接打印出来，应该使用echo print_r($arr).可以使用echo $arr\[2]\[1]将x打印出来。  
添加元素可以直接$arr\[3] = “newItem”;  
count(), max(), min(),  
sort(), rsort()反序排列。排序过后无序数组已经不存在。  
implode("" \* ", arr) 将每个元素用\*隔开并形成一个字符串   
explode(" \* ", arr) 将字符串转化成一个数组，将\*当成不同元素箭的分隔符  
in_array(19, arr) 查找19是否存在于数组中  
current() 返回指针当前所指向的对象，可以使用prev(), next(), reset(), end()移动指针  
13.	关联数组方法   
```$assoc = array(“first_name” => “Kevin”, “last_name” => “Steve”);```
使用echo $assoc[“first_name”]将Kevin打印出来。
14.	布林值方法  
is_bool()
15.	列表  
```list($var1, $var2, …)=```
16.	其他方法  
is_null(),    
isset() 是否赋值，null不算赋值  
empty(), “0”返回true  
\_\_FILE__ 返回当前文件路径（不是一个方法）  
dirname()表示当前文件所在文件夹的路径  
17.	类型转换  
```getType(); ```
```settype($var, “integer”);```直接转换了类型  
```$var2 = (string) $var;```并未直接转换该变量的类型  
18.	常量  
定义常量不能使用=，需要使用如下方法  
define(“MAX-WIDTH”, 980);  
在定义后不能重新定义 
19.	比较  
==检测值是否相等，===检测是否为同一对象  
20.	循环  
```foreach($var1 as $var2){ } ```
21.	方法  
```
function name(){ 
	return $var;
}
```
e.g.   
```function paint($color = “red”){ } 表示如果调用该方法时没有传参数，则默认是red```
22.	调试   
var_dump($var); 显示变量类型和值，不需要echo  
get_defined_vars(); 以数组形式显示开发人员和程序员自己已经定义的变量，需要使用print_r(get_defined_vars() );将它们打印出来;  
debug_backtrace(); 在方法内调用var_dump(debug_backtrace() );会返回该方法的相关信息  
可以使用Xdebug，DBG等第三方软件帮忙调试  

# 写码技巧：
1.	将需要重复调用的语句写到一个php文件里面，然后可以通过
```
<?php require_once ('../initialize.php'); ?>
<?php require ('../initialize.php'); ?>
<?php include ('../initialize.php'); ?>
<?php include_once ('../initialize.php'); ?>
```
等方式调用。
2.	在超链接网址时使用\<a href="xxxx=\<?php echo urlencode(‘’); ?>">Link\</a>\<br />有助于解决特殊字符问题  
3.	方法内调用全局变量的时候，在方法内写global加变量名  
4.	可以采用<script>info()/alert()/warning();</script>等方式在网页上弹出消息  
5.	http header如果想返回一个404错误可以这么写  
```
header($_SERVER[“SERVER_PROTOCOL”] . “ 404 Not Found”);
exit();
```
6.	Page redirects are sent in header. Headers are sent before page data. Header changes must be made before any HTML output.  
可使用header(“Location: index.php”);  
7.	启用output buffer  
Turn output buffering on in php.ini  
Use ob_start() when code may be ported to other servers.   
8.	链接数据库  
```
$connection = mysqli_connect($host, $user, $password, $database);
mysqli_close($connection);
```
9.	查询数据库
```
mysqli_query($connection, $query);
mysqli_free_result($result_set); //release returned data
mysqli_fetch_row();
mysqli_fetch_assoc(); //keys are column name
mysqli_fetch_array();
mysqli_num_rows($result); //return the number of rows
mysqli_connect_errno(); //return the number of error
mysqli_connect_error(); //return strings that describe the error
mysqli_insert_id(); //return the latest inserted id 
```
10.	Type juggling  
String vs. null: converts null to ""  
Number vs. other: converts other to number   
11.	Sanitize data //一定程度上防止黑客用恶意请求的方式盗取数据  
```
addshlashes($str);
mysqli_real_escape_string($conn, $str);  
$sql = “SEELECT * subjects”; 
$sql .= “ WHERE id=” . (int) $($id);
```
