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
  
  

