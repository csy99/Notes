# Basic

## Variable

Convention: all caps for system variables, lower case for user defined variables. 

```shell
echo $BASH
echo $BASH_VERSION
echo $HOME
echo $PWD
```

For user defined var, see the code below. Note that there is no space between var name and "=" and its value. 

```shell
name=Mark
echo The name is $name
```

### Array

Separate each element with space, not comma.  

```shell
os=('ubuntu' 'windowns' 'mac')
os[3]='kali'  # add/set an element
echo "${os[@]}"  # print all elements
echo "${!os[@]}"  # print the indices (may not be consecutive)
echo "${#os[@]}"  # print the length of array
unset os[3]  # remove an element 

# string
string=dasfsd
## the following all prints the full string
echo "${os[@]}"
echo "${os[0]}"
echo "${os[1]}"
```

## Logical Operator

**And**

```shell
age=25
if [ $age -gt 18 ] && [ $age -lt 30 ]
# Alternatively, we can use
# if [ $age -gt 18 -a $age -lt 30 ]
# if [[ $age -gt 18 && $age -lt 30 ]]
then 
	echo "valid age"
fi
```

**Or**

replace the second with -o

## Arithmetic Operation

Notice the space between parenthesis and condition. 

```shell
# integer
num1=10
num2=5
echo $(( num1+num2 ))
echo $(expr $num1 + $num2 )  # this does not work for *

# float
num1=20.5
num2=5
echo "20.5/5" | bc #4
echo "scale=2;20.5/5" | bc #4.10
echo "$num1+$num2" | bc
echo "scale=3;sqrt($num1)" | bc -l  # -l calls the library
echo "scale=3;3^3" | bc -l
```

## Control Flow

### IF

Note the space between condition and "[]".

**integer comparison**

-eq, -ne, -gt, -ge, -lt, -le: typical with [condition]

<, <=, >, >=: typical with ((condition))

**string comparison**

== (equivalent to =), !=: typical with [condition]

 <, >, -z- (string is null): typical with [[condition]]

```shell
# integer comparison 
cnt=10
if [ $cnt -eq 10 ]
then 
	statement
fi

# string comparison
word=abc
if [ $word == 'abc' ]
then 
	statement
elif [[ $word < 'zzz' ]]
then
	statement
else
	statement
fi
```

### Case

[a-z], [A-Z], [0-9]

? matches one letter character

if [a-z] also match upper case letter, execute `LANG=C`.

```shell
case expression in
	pattern1 )
		statement ;;
	pattern2 )
		statement ;;
	* ) # * matches any strings
		echo "unknown" ;;
esac
```

### While

Pattern. 

```shell
while [ condition ]
do 
	command1
	command2
done
```

We can sleep and open terminal. 

```shell
n=1
while (( n <= 10))
do
	echo "$n"
	(( n++ ))
	sleep 1
	gnome-terminal & 
	xterminal
done
```

### Until 

Equivalent to while (!condition).

### For

```shell
for var in 1 2 3 4 5  # in {1..5..1}, start, end, step
for var in file1 file2 file3
for (( exp1; exp2; exp3 ))
```

### Select 

Iterate the list and give user a menu-like prompt. 

```shell
select name in mark john tom ben
do 
	echo "$name selected"
done
```

Usually combine with case. 

```shell
select name in mark john tom ben
do 
	case $name in 
	mark )
		echo "mark selected" ;;
	john )
		echo "john selected" ;;
	tom )
		echo "tom selected" ;;     
    * )
    	echo "please provide no.1-4"
    esac
done
```

## Function

When we call the function, no parenthesis is needed. 

```shell
function name(){
	command
}

function print(){
	echo $1 $2 $3
}
print Hello World Again
```

Default, the variable is global. If we want to declare local variable, use local keyword. 

```shell
function print(){
	name=$1
	# local name=$1
	echo "the name is $name"
}
name="Tom"
print Max
echo "The name is $name: After"
# will be Tom if local keyword is used; Max if not. 
```

```shell
usage() {
	echo "You need to provide an argumet"
	echo "usage: $0 file_name"
}
file_exist() {
	local file="$1"
	[[ -f "$file" ]] && return 0 || return 1
}
[[ $# eq 0 ]] && usage
if ( file_exist "$1" )
then
	echo "file found"
else
	echo "file not found"
fi
```



# IO

### read inputs

```shell
echo "Enter Personal Info"
read name age email 
echo "name: $name, age: $age, email: $email"
```

If we want user to enter info in the same line. 

```shell
read -p 'username: ' user_var
echo 'username is' $user_var
```

If we want to silence the input (not displaying it).

```shell
read -sp 'password:' pw
echo  # let next message appear in a new line
echo 'password is' $pw
```

We can save all inputs in an array. 

```shell
echo "enter names:"
read -a names 
echo "Names include: ${names[0]}, ${names[1]}"
```

Using while loop. 

```shell
while read line
do 
	echo $line
done < content.txt

# an alternative way
cat content.txt | while read p
do
	echo $p
done
```



```shell
while IFS=' ' read -r line  # -r will skip escape sign
do 
	echo $line
done < content.txt
```



### Pass Arguments

Print the argument. '$0' represent the shell script name. 

```shell
echo $1 $2 $3
```

We can store the arguments in an array. The shell script name will not be parsed into the array, thus \${args[0]} is in fact \$1.

```shell
args=("$@")
echo ${args[0]}
echo $@  # print out all arguments 
echo $#  # print out the number of arguments 
```

### File test operators

-e check whether the file exists

```shell
echo -e "enter filename: \c"
read filename
if [-e $filename]
then 
	echo "$filename found"
else
	echo "$filename not found"
fi
```

### Append output

-f check whether the path is a file 

-w check write permission

\> overwrite a file

\>\> append to a file

```shell
echo -e "enter filename: \c"
read filename
if [ -f $filename ]
then 
	if [ -w $filename ]
	then 
		echo "I want to append"
		cat >> $filename
	else
		echo "no permission"
	fi
fi
```

