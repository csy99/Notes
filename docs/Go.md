# Intro
compiled, statically typed; compiled executables OS specific.  
some OO features:  
custom interface, types, structs  
not OO features:   
no inheritance, no operator/method overloading, no structured exception handling, no implicit numeric conversion

# Cmd Tool
gofmt demo.go   // only format in the cmd line  
gofmt -w demo.go  // format the original file  
go run demo.go  // compile and run, but does not save the executable file  
go build demo.go  // create new persistent executable file  
go install  

code demo
```
package main 
import (
“fmt”
“strings”
“bufio”
“os”
“strconv”
)

func main() {
	fmt.Println(strings.ToUpper(“Hello”))
	str1 := “The”
	str2 := “World”
	// automatically add white space
	stringLength, err := fmt.Println(str1, str2)  
	if err == nil {  // no error
		fmt.Println(“len is”, stringLength)
	}
	// if err is not used, compile error (var not used)
	// solution: use _ to notify that var should not be addressed

	aNumber := 42
	isTrue := true
	fmt.Printf(“value is %v, and boolean is %v\n”, aNumber, isTrue)
	fmt.Printf(“float value is %.2f \n”, float64(aNumber))
	fmt.Printf(“Data types is %T, %T, %T \n”, str1, aNumber, isTrue)
	myString := fmt.Sprintf(“Data types is %T, %T, %T \n”, str1, aNumber, isTrue)

	// read from console
	var s string
	fmt.Scanln(&s)  // only one word

	// read a line
	reader := bufio.NewReader(os.Stdin)
	str, _ := reader.ReadString(‘\n’)
	fmt.Println(str)

	// read a number 
	str, _ := reader.ReadString(‘\n’)
	f, err := strconv.ParseFloat(strings.TrimSpace(str), 64)
	if err != nil {
		fmt.Println(err)
	}
}
```  
# Variable Declaration  
### explicitly typed declarations
var myvar int = 42 / const myvar int = 42
### implicitly typed declarations
myvar := 42 / const myvar = 42

# Function Declaration 
```
func add(values ... int) {  // receive a list of int
	sum := 0
	for i := range values {
		sum += values[i]
	}
}
func FullName(f, l string) (string, int) {  // parenthesis 1: parameter; 2: return value
	full := f + “ “ + l
	length := len(full)
	return full, length
}
func FullName2(f, l string) (full string, length int) {  // specify return variable 
	full = f + “ “ + l  // declared in function declaration, no “:=”
	length = len(full)
	return  // do not specify again
}
```
<br/><br/>
# Common Packages
## strings pkg
```
strings.EqualFold(a, b)  // ignore case comparison  
strings.Contains(a, “ff”)  
```
## math pkg
numeric types do not implicitly convert, can’t add int to a float  
```
import “math/big”  
var b1, b2, b3, bigSum big.Float  
b1.SetFloat64(23.5)  
b2.SetFloat64(63.5)  
b3.SetFloat64(53.5) 
bigSum.Add(&b1, &b2). Add(&bigSum, &b3)
fmt.Printf(“BigSum = %.10g\n”, &bigSum)
circleR := 15.5
cC := circleR * math.Pi
```    
    
# Memory Allocation
allocated automatically and deallocated automatically by garbage collector  
- new(): allocates but does not initialize mem, results in zeroed storage but returns a memory address  
- make(): allocates and initializes memory, returns a memory address

```
var m map[string]int
m[“key”] = 43  // will result panic, assignment to entry in nil map 
m := make(map[string]int)
m[“key”] = 43  // successfully assignment 
```

# Complex Types and Collections
## pointer
```
var p *int
if p != nil {
fmt.Println(“Value of .p:”, *p) 
} else {
fmt.Println(“p is nil”) 
}
var v int  = 42
p = &v
```
## array
```
var colors [3] string
colors[0] = “red”
colors[1] = “2red”
colors[2] = “3red”
var numbers = [5]int {5,3,1,2,4}  // assign when init
```

## slice
```
var colors = []string {“red”, “green”, “blue”}
colors  = append(colors, “gray”)  // return a new reference
colors = append(colors[1:len(colors)])   // remove the first element
numbers := make([]int, 5, 5)
numbers[0] = 1
numbers[1] = 3
numbers[2] = 5 
numbers[3] = 7
numbers[4] = 9
numbers = append(numbers, 235)  // capacity is now changed to 10
sort.Ints(numbers)
```
## map
```
states := make(map[string] string)  // this is unordered
states[“wa”] = “Washington”
states[“wi”] = “Wisconsin”
states[“ca”] = “California”
c_ := states[“ca”]
delete(states, “wi”)  // delete a certain element

for k, v := range states {
	printf(“%v: %v”, k, v)
}
keys := make([]string, len(states))
i := 0
for k := range states {
	keys[i] = k
	i++
}
sort.Strings(keys)
println(“\n sorted”)
for i := range keys {
	println(states[keys[i]])
}
```

## struct
```
type Dog struct {
	Breed string
	Weight int
}
poodle := Dog {“poodle”, 34}
printf(“%+v\n”, poodle)
``` 

# Program Flow
```
defer println(“fff”)  // wait till others are executed, in LIFO order!
no while loop in Go
goto endofprogram  // jump to the label
```
```
switch myvar := rand.Intn(6); myvar {  // can place the assignment before checking
	case 1:
		…
	case myvar > 4:  // can do the checking like if 
		…
	default:  // no need to add “break”; add “fallthrough” to behave the same as C
		…
}
```

# Inheritance
```
type Animal interface {
	Speak() string
}
type Dog struct {  // no “implements” in Go, use function to infer
Sound string 
}
func (d Dog) printSound() string{
 	println(d.Sound)
}
func (d Dog) Speak() string{
 	return “arf”
}
type Cat struct {  // no “implements” in Go, use function to infer
}
func (c Cat) Speak()  string{
 	return “meow”
}

poodle := Animal(Dog{“woof”})  // cast 
poodle.printSound ()   // print out woof
animals := []Animal{Dog{‘a’}, Cat{}}
for _, animal := range animals {
	println(animal.Speak())
}
```
# handle error
```
f, er := os.Open(“file.txt”)
if er == nil {
}
myError := errors.New(“my error”)
```
# IO
```
func checkError(err error) {
	if err != nil {
		panic(err)
	}
}

// write to file method 1
file, err := os.Create(“./fromString.txt”)
checkError(err)
defer file.Close()
leng, err := io.WriteString(file, content)
checkError(err)
printf(“Finish with file of %v chars”, leng)

// write to file method 2
bytes := []byte(content)
ioutil.WriteFile(“./fromBytes.txt”, bytes, 0644)

// read from file 
content, err := ioutil.ReadFile(filename)  // the first return value is []byte
result := string(content)
```


# Traverse the File Path 
```
processPath(path string, info os.FileInfo, err error)  error {  // parameter type is fixed
	if err != nil { return err }
	if path != “.” {
		if info.IsDir() {
			println(“directory:”, path)
		} else {
			println(“file:”, path)
		}
	}
return nil
}

root, _ := filepath.Abs(“.”)
err := filepath.Walk(root, processPath)  // walk from file path
```

# Read from Web + Parse from json
```
type Tour struct {  // depends on the json file you receive
	Name, Price string
}
func tourFromJson(content string) []Tour {
	tours := make([]Tour, 0, 20)
	decoder := json.NewDecoder(strings.NewReader(content))
	_, err := decoder.Token()
	checkError(err)
	var tour Tour 
	for decoder.More() {
		err := decoder.Decode(&tour)
		checkError(err)
		tours = append(tours, tour)
	}
	 return tours
}
func contentFromServer(url string) string {
resp, err := http.Get(url)
defer resp.Body.Close()
bytes, err := ioutil.ReadAll(resp.Body)
return  string (bytes)
}

url := “http://”
content := contentFromServer(url)
tours := toursFromJson(content)
```

# Create a Http Server
```
type Hello struct {}
func (h Hello) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, “…”)
}
var h hello
err := http.ListenAndServe(“localhost: 4000”, h)
```

