# Cmd

## ls
```ls -l``` # list mode  
```ls -la``` # list + all hidden files  
```ls -lh``` # human readable size  

## help
```man pwd```  # open manual  
```man -h```  # find for help  
```grep --help```  # in more details  
## remove
```rm -r dir1```  recursively delete dir1 and all dirs and files under dir1  

## file contents

### print contents
```cat a.txt b.txt``` # print in sequence  
```cat < a.txt```  # read from stdin
```head a.txt```  # first 10 line  
```tail a.txt```  # last 10 line  
```head a.txt -n 5``` # first 5 line  
```less a.txt```  # read only version of vi
### search
```/ str```  # press n to find next, shift+n to find previous one  
```grep -n sim  ```
### word count
```wc ```

## IO

### rediret
```> ```# rediret to a file
```>> ```# append to a file
### pipe
```man less | grep sim | grep That > that.txt```  
## player

### conversion
```ffmpeg -i bad_appple.mp4 bad_appple_h264.mp4```  
### player
```mplayer -vo caca -quiet bad_apple_h264.mp4```  

# script

## execute python3 script
add following code to the script:  
```
#!/usr/bin/env python3
```
run on cmd line:  
```> chmod +x my_echo.py``` # add execution authorization  
```PATH=$PATH:$PWD``` # even in other dir can still execute it  
```which python3``` # the root dir for python3
## change mode
```chmod 740 foo``` owner, group, others; default 640; read=4, write=2, execute=1  

# Event Processing
"/dev/null"是设备文件，丢弃所有写入数据但返回写入成功。
```
cat *.pgn > /dev/null是设备文件，丢弃所有写入数据但返回写入成功。
```
使用time命令。  
读取速度测试，文件处理速度上线。
## Create Pipe
```
cat *.pgn | grep "result" | sort | uniq -c
```
- sort  
会把数据读入内存，若发不下则写入临时文件。  
time: O(nlogn), space: O(n)  
- uniq -c  
统计每个独立行出现次数，仅对已排序文本有效。
time: O(n), space: O(1)  
## AWK
domain-specific language. 
```
cat *.pgn | grep "result" | awk '{split($0, a, "-"); res = 
substr(a[1], length(a[1]), 1); if (res == 1) white++;
if (res == 0) black++;} END {print white+black, white, black}'
```
$0输入行  
split( , , "-")按-分割  
substr(a\[1], length(a\[1]), 1)取出最后一个字符
time: O(n), space: O(1)  
## Parellel Programming
管道中命令并行执行。  
- xargs -n1 -P8 每一次最多取1个参数，最多8个命令同时执行   

  

# 进程管理

### Stress

给系统增加负载，进行压力测试

```shell
-t/--timeout N # N秒后超时
-c/--cpu N # 孵化N个worker，死循环运行sqrt()
-i/--io N # 孵化N个worker，死循环运行sync()
-m/--vm N # 孵化N个worker，死循环运行malloc()&free()
-d/--hdd N # 孵化N个worker，死循环运行write()&unlink()

stress -c 16 
```

### Top

显示或更新排序过的进程信息，默认按照cpu占用率排序。

### PS

显示进程状态。默认只显示当前用户有控制终端的进程

```shell
ps aux # 显示所有进程
ps -l # 显示pid和ppid
ps aux | grep Chrome | wc -l
```

### Kill

```shell
kill -signal_number/-signal_name PID
kill PID # 空格分隔多个进程
kill -9/-KILL PID # 强力杀进程
killall bash # 按照名字终止进程
```

### keyboard

ctrl+C是发送SIGINT中断信号。

ctrl+Z发送SIGTSIP停止信号。进程还存在，放到后台挂起，打开的端口仍然被占用。

### 前后台

```shell
& # 在后台运行进程（加在命令最后）
jobs # 显示从当前终端启动的命令 
fg %1 # 把后台进程放到前台，后面的数字是jobs中的数字
bg # 继续被挂起的后台进程
```

