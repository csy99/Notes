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

  

