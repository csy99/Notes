# 简介

软件工程中，设计模式是对软件设计中普遍存在的各种问题所提出的解决方案。

编写代码，需要解决耦合性、内聚性、可维护性、可扩展性、重用性、灵活性的问题。

低耦合：该是哪个模块的错误就在哪个模块报错

高内聚：

代码重用性：相同功能的代码，不用多次编写

可读性：编程规范性，便于其他程序员的阅读和理解

可扩展性：需要增加新的功能时成本低

可靠性：增加新的功能后，对原来的功能没有影响

### 七大设计原则核心思想

1. 单一职责
2. 接口隔离
3. 依赖倒转
4. 里式替换
5. 开闭原则ocp
6. 迪米特法则
7. 合成复用原则

#### 单一职责 single responsibility

一个类只负责一项职责，降低类的复杂度。

```java
public class SingleResp{
    psvm() {
        // 1
        Vehicle vehicle = new Vehicle();
        vehicle.run("摩托车");
        vehicle.run("汽车");
        vehicle.run("飞机");
        
        // 2
        RoadVehicle roadVehicle = new RoadVehicle();
        roadVehicle.run("摩托车");
        roadVehicle.run("汽车");
        AirVehicle airVehicle = new AirVehicle();
        airVehicle.run("飞机");

        // 3
        Vehicle3 vehicle3 = new Vehicle3();
        vehicle3.run("汽车");
		vehicle3.runWater("轮船");
		vehicle3.runAir("飞机");
	}
}

//方案 1 的分析
// 在方式1的run方法中，违反了单一职责原则(海陆空都管)
class Vehicle {
    public void run(String vehicle) { 		
        System.out.println(vehicle + " 在公路上运行");
    }
}

//方案 2 的分析
// 遵守单一职责原则，但是这样做的改动很大，即将类分解，同时修改客户端
class RoadVehicle {
    public void run(String vehicle) { 
        System.out.println(vehicle + "公路运行");
    }
}


class AirVehicle {
    public void run(String vehicle) { 
        System.out.println(vehicle + "天空运行");
    }
}

//方案 3 的分析
// 在方法级别上，仍然是遵守单一职责
class Vehicle3 {
    public void run(String vehicle) {
        System.out.println(vehicle + " 在公路上运行....");
    }


    public void runAir(String vehicle) { 
        System.out.println(vehicle + " 在天空上运行....");
    }


    public void runWater(String vehicle) { 
        System.out.println(vehicle + " 在水中行....");
    }
}
```

#### 接口隔离原则 Interface Segregation Principle

客户端不应该依赖它不需要的接口，即一个类对另一个类的依赖应该建立在最小的接口上。

```java
public class Segregation1 {
    public static void main(String[] args) {

    }
}

//接口
interface Interface1 {
	void operation1(); 
    void operation2();
	void operation3(); 
    void operation4(); 
    void operation5();
}


class B implements Interface1 { 
    public void operation1() {
		System.out.println("B 实现了 operation1");
	}
    public void operation2() {
        System.out.println("B 实现了 operation2");
    }
    public void operation3() {
        System.out.println("B 实现了 operation3");
    }
    public void operation4() {
    	System.out.println("B 实现了 operation4");
    }
    public void operation5() {
    	System.out.println("B 实现了 operation5");
    }
}

class D implements Interface1 { 
    public void operation1() {
		System.out.println("D 实现了 operation1");
	}
    public void operation2() {
    	System.out.println("D 实现了 operation2");
    }
    public void operation3() {
    	System.out.println("D 实现了 operation3");
    }
    public void operation4() {
    	System.out.println("D 实现了  operation4");
    }
    public void  operation5()  { 
        System.out.println("D 实现了  operation5");
    }
}

//A 类通过接口Interface1依赖(使用) B 类，但是只会用到 1,2,3 方法
class A { 
    public void depend1(Interface1 i) { 
        i.operation1();
    }
    public void depend2(Interface1 i) { 
        i.operation2();
    }
    public void depend3(Interface1 i) { 
        i.operation3();
    }
}

//C 类通过接口Interface1依赖(使用) D 类，但是只会用到 1,4,5 方法
class C { 
    public void depend1(Interface1 i) { 
        i.operation1();
    }
    public void depend4(Interface1 i) { 
        i.operation4();
    }
    public void depend5(Interface1 i) { 
        i.operation5();
    }
}
```

修改后的代码，将接口拆分。

```java
public class Segregation1 {
    public static void main(String[] args) {
        // 使用一把
        A a = new A();
        a.depend1(new B()); // A 类通过接口去依赖 B 类
        a.depend2(new B());
        a.depend3(new B());

        C c = new C();
        c.depend1(new D()); // C 类通过接口去依赖(使用)D 类
        c.depend4(new D());
        c.depend5(new D());
    }
}

// 接 口 1
interface Interface1 { 
    void operation1();
}

// 接 口 2
interface Interface2 { 
    void operation2();
	void operation3();
}

// 接 口 3
interface Interface3 { 
    void operation4();
	void operation5();
}


class B implements Interface1, Interface2 {
    public void operation1() {
    	System.out.println("B 实现了 operation1");
    }

    public void operation2() {
    	System.out.println("B 实现了 operation2");
    }

    public void operation3() {
    	System.out.println("B 实现了 operation3");
    }
}


class D implements Interface1, Interface3 { 
    public void operation1() {
		System.out.println("D 实现了  operation1");
	}

    public void  operation4()  { 
        System.out.println("D 实现了  operation4");
    }

    public void operation5() {
    	System.out.println("D 实现了 operation5");
    }
}

// A 类通过接口 Interface1,Interface2 依赖(使用) B 类，但是只会用到 1,2,3 方法
class A { 
    public void depend1(Interface1 i) { 
        i.operation1();
    }

    public void depend2(Interface2 i) { 
        i.operation2();
    }

    public void depend3(Interface2 i) { 
        i.operation3();
    }
}

// C  类通过接口 Interface1,Interface3  依赖(使用) D 类，但是只会用到 1,4,5 方法
class C { 
    public void depend1(Interface1 i) { 
        i.operation1();
    }

    public void depend4(Interface3 i) { 
        i.operation4();
    }

    public void depend5(Interface3 i) {
    	i.operation5();
    }
}
```

#### 依赖倒转原则 Dependence Inversion Principle

高层模块不应该依赖底层模块，二者都应该依赖其抽象。抽象不应该依赖细节。中心思想是面向接口编程。使用接口的目的是制定好规范。

依赖传递三种方式：通过接口传递，通过构造方法，通过setter方式传递。

```java
public class DependencyInversion {
    public static void main(String[] args) { 
        Person person = new Person(); 
        person.receive(new Email());
    }
}

class Email {
    public String getInfo() {
        return "电子邮件信息: hello,world";
    }
}

//完成 Person 接收消息的功能
//方式 1 分析
//1. 简单，比较容易想到
//2. 如果我们获取的对象是 微信，短信等等，则新增类，同时 Person 也要增加相应的接收方法

class Person {
    public void receive(Email email) { 			
        System.out.println(email.getInfo());
    }
}

```

解决思路：引入一个抽象的接口 IReceiver, 表示接收者, 这样 Person 类与接口 IReceiver 发生依赖。因为 Email, WeiXin 等等属于接收的范围，他们各自实现 IReceiver 接口就 ok,  这样我们就符合依赖倒转原则。

```java
public class DependencyInversion {
    public static void main(String[] args) {
        //客户端无需改变
        Person person = new Person(); 
        person.receive(new Email());
        person.receive(new WeiXin());
    }
}

//定义接口
interface IReceiver { 
    public String getInfo();
}

class Email implements IReceiver { 
    public String getInfo() {
		return "电子邮件信息: hello,world";
	}
}
//增加微信
class WeiXin implements IReceiver { 
    public String getInfo() {
        return "微信信息: hello,ok";
	}
}

//方式 2
class Person {
//这里我们是对接口的依赖
    public void receive(IReceiver receiver ) { 
        System.out.println(receiver.getInfo());
    }
}
```

#### 里式替换原则

1)    继承包含这样一层含义：父类中凡是已经实现好的方法，实际上是在设定规范和契约，虽然它不强制要求所有的子类必须遵循这些契约，但是如果子类对这些已经实现的方法任意修改，就会对整个继承体系造成破坏。

2)    继承在给程序设计带来便利的同时，也带来了弊端。比如使用继承会给程序带来侵入性，程序的可移植性降低， 增加对象间的耦合性，如果一个类被其他的类所继承，则当这个类需要修改时，必须考虑到所有的子类，并且父类修改后，所有涉及到子类的功能都有可能产生故障

在使用继承时，遵循里氏替换原则，在子类中尽量不要重写父类的方法。原来的父类和子类都继承一个更通俗的基类。如果需要发生关系，可以采用聚合、组合。

#### 开闭原则 Open Closed Principle

一个软件实体应该对扩展开放（对提供方），对修改关闭（对使用方）。用抽象构建框架，用实现扩展细节。当软件需要变化时，尽量通过扩展软件实体的行为来实现，而不是通过修改。

#### 迪米特法则

也叫最少知道原则。一个对象应该对其他对象保持最少的了解。类和类关系越密切，耦合度越大。

直接的朋友：耦合的方式包括依赖、关联、组合、聚合等。出现了成员变量，方法参数，方法返回值中的类是直接的朋友。出现在局部变量中的类不是直接的朋友。

#### 合成复用原则

尽量使用合成、聚合的方式，而不是使用继承。



# UML类图

<img src="./pics/04design_uml.jpg">

### 类之间的关系

#### 依赖

在类中用到了对方。

#### 泛化

实际上就是继承，依赖关系的特例。

#### 实现

实现了接口。

#### 关联

依赖关系的特例。具有导航性，即双向或单向关系。有单向一对一关系，双向一对一关系。

#### 聚合

整体与部分可以分开，是关联关系的特例。

```java
public class Computer {
    private Mouse mouse;
    Private Monitor monitor;
    
    public void setMouse(Mouse m) {
        mouse = n;
    }
    
    public void setMonitor(Monitor m) {
        monitor = m;
    }
}
```

#### 组合

整体与部分不能分开，是关联关系的特例。

```java
public class Computer {
    private Mouse mouse = new Mouse(); //鼠标和 computer 不能分离
	private Moniter moniter = new Moniter();//显示器和 Computer 不能分离
    public void setMouse(Mouse mouse) {
		this.mouse = mouse;
	}
	public void setMoniter(Moniter moniter) { 
        this.moniter = moniter;
	}
}
```



# 设计模式

## 创建型模式

### 单例模式

整个软件系统中一个类只能有一个对象，节省了系统资源。

java.lang.Runtime是经典的案例。

#### 饿汉式

构造器私有化（防止new），在类的内部创建对象，对外暴露一个静态公共对象。

**优点**

1. 类装载的时候完成实例化，避免了线程同步问题

**缺点**

1. 如果从始至终没有使用过，可能造成内存浪费

```java
// 写法1
class Singleton {
    private Singleton(){}
    
    private final static Singleton instance = new Singleton();
    
    public static Singleton getInstance() {
        return instance;
    }
}

// 写法2
class Singleton {
    private static Singleton instance;
    
    static {
        instance = new Singleton();
    }
    
    private Singleton() { }   
    
    public static Singleton getInstance() {
        return instance;
    }
}
```

#### 懒汉式

**优点**

1. 懒加载

**缺点**

1. 使用静态内部类无法传参

```java
// 线程不安全
class Singleton {
    private static Singleton instance;
    
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null)
            instance = new Singleton();
        return instance;
    }    
}

// 线程安全，效率较低
class Singleton {
    private static Singleton instance;
    
    private Singleton() {}
    
    public static synchronized Singleton getInstance() {
        if (instance == null)
            instance = new Singleton();
        return instance;
    }    
}

// 双重检查
class Singleton {
    private static volatile Singleton instance;
    
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (stance == null)
                    instance = new Singleton();
            }            
        }
        return instance;
    }    
}

// 静态内部类
class Singleton { 
    private Singleton() {}
    
    private static class SingletonInstance {
        private static final Singleton INSTANCE = new Singleton();
    }
    
    public static Singleton getInstance() {
        return SingletonInstance.INSTANCE;
    }    
}
```

#### 枚举

借助jdk1.5中添加的枚举来实现单例模式。

**优点**

1. 防止反序列化重新创建新的对象

**缺点**

1. 

```java
enum Singleton{
    INSTANCE; //属性
    public void sayOK() { 
        System.out.println("ok~");
    }
}
```

### 简单工厂

定义了一个创建对象的类，用它来封装实例化对象的行为。

在`java.util.Calendar`中`createCalendar()`有使用。

**意义**

放到一个类中统一管理和维护，达到和主项目的解耦。

**依赖抽象原则**

1. 创建对象实例，不要直接new，把这个动作放在一个工厂的方法中作为返回值
2. 不要继承具体类，继承抽象类或者实现接口
3. 不要覆盖基类中已经实现的方法

**优点**

1. 提高扩展性和维护性。

```java
public class SimpleFactory {
    //更加 orderType 返回对应的 Pizza 对象
    public Pizza createPizza(String orderType) {
    	Pizza pizza = null;
    	System.out.println("使用简单工厂模式"); 
        if (orderType.equals("greek")) {
    		pizza = new GreekPizza();
    		pizza.setName(" 希腊披萨 ");
    	} else if (orderType.equals("cheese")) { 
            pizza = new CheesePizza();
    		pizza.setName(" 奶酪披萨 ");
    	} else if (orderType.equals("pepper")) { 
            pizza = new PepperPizza();
    		pizza.setName("胡椒披萨");
    }
    return pizza;
}

class OrderPizza {
    //定义一个简单工厂对象
    SimpleFactory simpleFactory; 
    Pizza pizza = null;

    //构造器
    public OrderPizza(SimpleFactory simpleFactory) { 		
        setFactory(simpleFactory);
    }

    public void setFactory(SimpleFactory simpleFactory) {
        String orderType = ""; //用户输入的
        this.simpleFactory = simpleFactory; //设置简单工厂对象
        do {
            orderType = getType();
            pizza = this.simpleFactory.createPizza(orderType);
            //输出 pizza
            if(pizza != null) { //订购成功
            	pizza.prepare(); 
                pizza.bake();
            	pizza.cut();
            	pizza.box();
            } else {
            	System.out.println(" 订购披萨失败 "); 
                break;
            }
        } while(true);
    }
}
```

### 工厂方法模式

定义了一个创建对象的抽象方法，由子类决定要实例化的类。工厂方法模式将对象的实例化推迟到子类。

```java
public abstract class OrderPizza {
    //定义一个抽象方法，createPizza(),  让各个工厂子类自己实现
    abstract Pizza createPizza(String orderType);

    // 构造器
    public OrderPizza() { 
        Pizza pizza = null;
    	String orderType; //  订购披萨的类型
        do {
            orderType = getType();
            //抽象方法，由工厂子类完成
            pizza = createPizza(orderType); 
            //输出 pizza 制作过程
            pizza.prepare(); 
            pizza.bake();
            pizza.cut();
            pizza.box();
        } while (true);
    }
}

//还可以创建其他的OrderPizza，实现方法类似
public class BJOrderPizza extends OrderPizza {
    @Override
    Pizza createPizza(String orderType) {
        Pizza pizza = null; 
        if(orderType.equals("cheese")) {
            pizza = new BJCheesePizza();
        } else if (orderType.equals("pepper")) { 
            pizza = new BJPepperPizza();
        }
    }
}
```

### 抽象工厂

定义了一个interface用于创建相关或有依赖关系的对象簇，而无需指明具体的类。将简单工厂模式和工厂方法模式进行整合。

### 原型模式

实现clone的java类需要实现Cloneable接口。用原型实例指定创建对象的种类，通过拷贝这些原型，创建新的对象。

在Spring框架中创建bean有使用。

数据类型是引用数据类型的，浅拷贝进行引用传递，而不是值拷贝。

**优点**

1. 创建新对象比较复杂时，使用原型模式提高效率
2. 能够动态获取对象运行时的状态

**缺点**

1. 需要为每一个类配备一个克隆办法，对已有的类进行改造时需要修改其源代码，违背了ocp原则

```java
public class Sheep implements Cloneable { 
    private String name;
    private int age; 
    private String color;
    private String address = "蒙古羊";
    public Sheep friend; //默认是浅拷贝
    public Sheep(String name, int age, String color) { 
        super();
    	this.name = name; 
        this.age = age; 
        this.color = color;
    }
    
    @Override
    protected Object clone() {
        Sheep sheep = null; 
        try {
        	sheep = (Sheep)super.clone();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return sheep;
    }

}

```

深拷贝可以通过序列化和重写clone方法实现。

```java
public class DeepProtoType implements Serializable, Cloneable{

	public String name; //String 属 性
	public DeepCloneableTarget deepCloneableTarget;// 引用类型
	public DeepProtoType() { }

	//深拷贝 - 方式 1  使用 clone 方法
	@Override
	protected Object clone() throws CloneNotSupportedException {
        Object deep = null;
        //这里完成对基本数据类型(属性)和 String 的克隆
        deep = super.clone();
        //对引用类型的属性，进行单独处理
        DeepProtoType deepProtoType = (DeepProtoType)deep;
        deepProtoType.deepCloneableTarget	= (DeepCloneableTarget) deepCloneableTarget.clone();
		return deepProtoType;
}

	//深拷贝 - 方式 2 通过对象的序列化实现 (推荐) 
    public Object deepClone() {
        ByteArrayOutputStream bos = null; 
        ObjectOutputStream oos = null; 
        ByteArrayInputStream bis = null; 
        ObjectInputStream ois = null;

        try {
            //序列化
            bos = new ByteArrayOutputStream(); 
            oos = new ObjectOutputStream(bos);
            oos.writeObject(this); //当前这个对象以对象流的方式输出

            //反序列化
            bis = new ByteArrayInputStream(bos.toByteArray()); 
            ois = new ObjectInputStream(bis);
            DeepProtoType copyObj = (DeepProtoType) ois.readObject();
            return copyObj;
        } catch (Exception e) {
            return null;
        } finally {
            try {
                bos.close();
                oos.close();
                bis.close();
                ois.close();
            } catch (Exception e2) {
                System.out.println(e2.getMessage());
            }
        }
    }
}
```

### 建造者模式

也叫生成器模式。将复杂对象的建造过程抽象出来，是这个过程的不同实现方法可以构造出不同表现的对象。

在StringBuilder中使用了建造者模式。Appendable接口定义了多个append方法(抽象方法)，即Appendable是抽象建造者。AbstractStringBuilder已经是建造者，只是不能实例化。StringBuilder既充当了指挥者，也充当了具体的建造者。

product：具体的产品对象

builder：创建一个product对象的各个部件指定的接口或抽象类

concrete builder：实现接口，装配各个部件

director：构建一个使用builder接口的对象。创建一个复杂的对象。

**优点**

1. 客户端不需要知道产品内部细节
2. 可以精细控制产品创建过程
3. 不需要改变类库代码，符合ocp原则

**缺点**

1. 产品内部变化过于复杂，不适合使用

```java
public class Client {
    public static void main(String[] args) {
        //盖普通房子
        CommonHouse commonHouse = new CommonHouse();
        //准备创建房子的指挥者
        HouseDirector houseDirector = new HouseDirector(commonHouse);
        //完成盖房子，返回产品(普通房子)
        House house = houseDirector.constructHouse();
        System.out.println("--------------------------");
        //盖高楼
        HighBuilding highBuilding = new HighBuilding();
        //重置建造者
        houseDirector = new HouseDirector(highBuilding);
        //完成盖房子，返回产品(高楼) 
        houseDirector.constructHouse();
    }
}

public class CommonHouse extends HouseBuilder {
    @Override
    public void buildBasic() {
    	System.out.println(" 普通房子打地基 5 米 ");
    }

    @Override
    public void buildWalls() {
    	System.out.println(" 普通房子砌墙 10cm ");
    }

    @Override
    public void roofed() {
    	System.out.println(" 普通房子屋顶 ");
    }
}

// 产 品 ->Product 
public class House {
    private String baise; 
    private String wall; 
    private String roofed; 
    // 省略getter & setter
}

// 抽象的建造者
public abstract class HouseBuilder {
	protected House house = new House();
	//将建造的流程写好, 抽象的方法
	public abstract void buildBasic(); 
    public abstract void buildWalls(); 
    public abstract void roofed();

    //建造房子好， 将产品(房子) 返回
    public House buildHouse() { 
        return house;
    }
}

public class HouseDirector {
    HouseBuilder houseBuilder = null;

    //构造器传入 houseBuilder
    public HouseDirector(HouseBuilder houseBuilder) { 
        this.houseBuilder = houseBuilder;
    }

    //如何处理建造房子的流程，交给指挥者 
    public House constructHouse() {
    	houseBuilder.buildBasic(); 
        houseBuilder.buildWalls(); 
        houseBuilder.roofed();
    	return houseBuilder.buildHouse();
    }
}
```

## 结构型模式

### 适配器模式

将某个类的接口转换成客户端期望的另一个接口表示，增强兼容性。三种命名是根据src以怎么样的形式给到Adapter来命名的。

在SpringMVC框架中HandlerAdapter有应用。

**优点**

1. 客户端不知道被适配者，是解耦的

#### 类适配器模式

继承src类，实现dst类接口。

**缺点**

1. Java 是单继承机制，所以类适配器继承 src 类要求 dst 必须是接口，有一定局限性
2. src 类的方法在 Adapter 中都会暴露出来，增加了使用的成本

```java
//被适配的类
public class Voltage220V {
    //输出 220V 的电压
    public int output220V() { 
        int src = 220;
    	System.out.println("电压=" + src + "伏");
    	return src;
    }
}

//适配接口
public interface IVoltage5V { 
    public int output5V();
}

//适配器类
public class VoltageAdapter extends Voltage220V implements IVoltage5V {

    @Override
    public int output5V() {
    	int srcV = output220V();
    	int dstV = srcV / 44 ; //转成 5v 
        return dstV;
    }
}
```

#### 对象适配器模式

对象适配器和类适配器其实算是同一种思想，只不过实现方式不同。根据合成复用原则，使用组合替代继承， 所以它解决了类适配器必须继承 src 的局限性问题，也不再要求 dst必须是接口。

```java
//适配器类
public class VoltageAdapter implements IVoltage5V {
	private Voltage220V voltage220V; // 关联关系-聚合
	//通过构造器，传入一个 Voltage220V 实例
	public VoltageAdapter(Voltage220V voltage220v) {
		this.voltage220V = voltage220v;
	}
    
	@Override
	public int output5V() {
        int dst = 0;
        if(null != voltage220V) {
            int src = voltage220V.output220V();//获取 220V 电压
            System.out.println("使用对象适配器，进行适配~~"); 
            dst = src / 44;
            System.out.println("适配完成，输出的电压为=" + dst);
        }
		return dst;
	}
}
```

#### 接口适配器模式

当不需要全部实现接口提供的方法时，可先设计一个抽象类实现接口，并为该接口中每个方法提供一个默认实现(空方法)，那么该抽象类的子类可有选择地覆盖父类的某些方法来实现需求。

在Android中的属性动画ValueAnimator类可以通过addListener方法添加监听器。

```java
public interface Interface4 { 
    public void m1(); 
    public void m2(); 
    public void m3(); 
    public void m4();
}

public abstract class AbsAdapter implements Interface4 {
    //默认实现
    public void m1() {}
    public void m2() {}
    public void m3() {}
    public void m4() {}
}

public class Client {
    public static void main(String[] args) {
    	AbsAdapter absAdapter = new AbsAdapter() {
            //只覆盖需要使用的接口方法
            @Override
            public void m1() {
                System.out.println("使用了 m1 的方法");
            }
   		};
    	absAdapter.m1();
    }
}
```

### 桥接模式

将实现和抽象放在两个不同的类层次中，使这两个层次可以独立改变。基于类的最小设计原则。

![05design_bridge](./pics/05design_bridge.jpg)

1)   Client 类：桥接模式的调用者

2)   抽象类(Abstraction) :维护了 Implementor / 即它的实现类 ConcreteImplementorA.., 二者是聚合关系, Abstraction充当桥接类

3)   RefinedAbstraction : 是 Abstraction 抽象类的子类

4)   Implementor : 行为实现类的接口

5)   ConcreteImplementorA /B ：行为的具体实现类

6)   从 UML 图：这里的抽象类和接口是聚合的关系，其实调用和被调用关系

在JDBC中的Driver接口中有使用。MySQL, Oracle等的driver。

**优点**

1. 减少子类个数，降低系统管理和维护成本（解决类爆炸问题）

**缺点**

1. 需要能够识别出系统中两个独立变化的维度，应用场景受限

### 装饰者模式

动态的将新功能附加到对象上。体现了ocp。

在Java的IO结构中FilterInputStream有使用。

<img src="./pics/06_deco.jpg">

**优点**

1. 减少子类个数，降低系统管理和维护成本（解决类爆炸问题）

```java
public class Coffee	extends Drink {

	@Override
    public float cost() {
    	return super.getPrice();
    }
}


public class CoffeeBar {
    public static void main(String[] args) {
        // 装饰者模式下的订单：2 份巧克力+一份牛奶的 LongBlack

        // 1.  点一份 LongBlack
        Drink order = new LongBlack();
        System.out.println("费用 1=" + order.cost());
        System.out.println("描述=" + order.getDes());

        // 2. order 加入一份牛奶
        order = new Milk(order);

        System.out.println("order 加入一份牛奶 费用 =" + order.cost());
        System.out.println("order 加入一份牛奶 描述 = " + order.getDes());

        // 3. order 加入一份巧克力


        order = new Chocolate(order);

        System.out.println("order 加入一份牛奶  加入一份巧克力	费 用 =" + order.cost());


        System.out.println("order 加入一份牛奶 加入一份巧克力 描述 = " + order.getDes());

        // 3. order 加入一份巧克力


        order = new Chocolate(order);

        System.out.println("order 加入一份牛奶  加入 2 份巧克力	费 用 =" + order.cost());
        System.out.println("order 加入一份牛奶 加入 2 份巧克力 描述 = " + order.getDes());


        System.out.println("===========================");


        Drink order2 = new DeCaf();

        System.out.println("order2 无因咖啡	费 用 =" + order2.cost());
        System.out.println("order2 无因咖啡 描述 = " + order2.getDes());
        order2               =                new                Milk(order2); System.out.println("order2 无因咖啡  加入一份牛奶	费 用 =" + order2.cost());
        System.out.println("order2 无因咖啡 加入一份牛奶 描述 = " + order2.getDes());
    }
}

public class Decorator extends Drink { 
    private Drink obj;

    public Decorator(Drink obj) { //组合 
        this.obj = obj;
    }

    @Override
    public float cost() {
    	// getPrice 自己价格
    	return super.getPrice() + obj.cost();
    }

    @Override
    public String getDes() {
    // TODO Auto-generated method stub
    // obj.getDes() 输出被装饰者的信息
    return des + " " + getPrice() + " && " + obj.getDes();
    }
}

public abstract class Drink { 
    public String des; // 描 述
    private float price = 0.0f; 
    
    public String getDes() {
    	return des;
    }
    public void setDes(String des) { 
        this.des = des;
    }
    public float getPrice() { 
        return price;
    }
    public void setPrice(float price) { 
        this.price = price;
    }

    //计算费用的抽象方法，子类来实现
    public abstract float cost();
}


public class Espresso extends Coffee {
    public Espresso() {
        setDes(" 意大利咖啡 "); 
        setPrice(6.0f);
    }
}


public class LongBlack extends Coffee {
    public LongBlack() {
        setDes(" longblack "); 
        setPrice(5.0f);
    }
}

public class ShortBlack extends Coffee{
    public ShortBlack() { 
        setDes(" shortblack "); 
        setPrice(4.0f);
    }
}

//具体的 Decorator， 这里就是调味品
public class Chocolate extends Decorator {
    public Chocolate(Drink obj) { 
        super(obj);
    	setDes(" 巧克力 ");
    	setPrice(3.0f); // 调味品 的价格
    }
}

public class Milk extends Decorator {
    public Milk(Drink obj) { 
        super(obj);
        setDes(" 牛 奶 "); 
        setPrice(2.0f);
    }
}


public class Soy extends Decorator{
    public Soy(Drink obj) { 
        super(obj);
        setDes(" 豆浆	"); 
        setPrice(1.5f);
    }
}
```

### 组合模式

也叫部分整体模式。创建了对象组的树形结构，将对象组合成树状结构以表示整体到部分的层次关系。

component可以是个类，也可以是个接口。composite是中间节点。leaf是叶子节点。

在HashMap中使用了组合模式，静态内部类Node是leaf，HashMap是composite，Map是component。

**优点**

1. 用户对单个对象和组合对象的访问具有一致性

```java
public class Client {
    public static void main(String[] args) {
    	//从大到小创建对象 学校
    	OrganizationComponent university = new University("清华大学", " 中国顶级大学 ");

    	//创建 学院
    	OrganizationComponent computerCollege = new College("计 算机学院", "计算机学院"); 
        OrganizationComponent infoEngineercollege = new College("信息工程学院", "信息工程学院");

        //创建各个学院下面的系(专业)
        computerCollege.add(new Department("软件工程", "软件工程不错 ")); 
        computerCollege.add(new Department("网络工程", "网络工程不错 "));
        computerCollege.add(new Department("计算机科学与技术", "计算机科学与技术是老牌的专业 "));
        infoEngineercollege.add(new Department("通信工程", "通信工程不好学")); 
        infoEngineercollege.add(new Department("信息工程", "信息工程好学"));

    	//将学院加入到 学校
        university.add(computerCollege); 		
        university.add(infoEngineercollege);
    	university.print(); 
        infoEngineercollege.print();
    }
}

public abstract class OrganizationComponent { 
    private String name; // 名 字
	private String des; // 说 明
 
	protected void add(OrganizationComponent organizationComponent) {
		//默认实现
		throw new UnsupportedOperationException();
}


	protected void remove(OrganizationComponent organizationComponent) {
		//默认实现
		throw new UnsupportedOperationException();
	}

    //构造器
    public OrganizationComponent(String name, String des) { 			super();
    	this.name = name;                                               this.des = des;
    }

	public String getName() { return name;}
	public void setName(String name) { this.name = name;}

	//方法 print,  做成抽象的,  子类都需要实现
	protected abstract void print();
}

//University 就是 Composite, 可以管理 College
public class University extends OrganizationComponent {
	List<OrganizationComponent> organizationComponents = new ArrayList<OrganizationComponent>();

    // 构造器
    public University(String name, String des) {
        super(name, des);
    }

    // 重 写 add @Override
    protected void add(OrganizationComponent oc) {
		organizationComponents.add(oc);
    }

    // 重 写 remove @Override
    protected void remove(OrganizationComponent organizationComponent) {
        organizationComponents.remove(organizationComponent);
    }

    // print 方法，就是输出 University  包含的学院
    @Override
    protected void print() {
        System.out.println("--------------" + getName() + "--------------");
        //遍历 organizationComponents
        for (OrganizationComponent oc: organizationComponents) { 
            oc.print();
        }
    }
}

public class College extends OrganizationComponent {
    //List 中 存放的 Department
    List<OrganizationComponent> organizationComponents = new ArrayList<OrganizationComponent>();

    // 构造器
    public College(String name, String des) { 
        super(name, des);
    }

    // 重 写 add @Override
    protected void add(OrganizationComponent organizationComponent) {
    //	将来实际业务中，Colleage和University的add不一定完全一样
    	organizationComponents.add(organizationComponent);
    }

    // 重 写 remove @Override
    protected void remove(OrganizationComponent organizationComponent) {
        organizationComponents.remove(organizationComponent);
    }

    // print 方法，就是输出 University  包含的学院
    @Override
    protected void print() {
        System.out.println("--------------" + getName() + "--------------");
        //遍历 organizationComponents
        for (OrganizationComponent oc: organizationComponents) { 
            oc.print();
        }
    }
}

public class Department extends OrganizationComponent {
    public Department(String name, String des) { 
        super(name, des);
    }

	//add , remove 就不用写了，因为他是叶子节点

    @Override
    public String getName() {
        return super.getName();
    }


    @Override
    public String getDes() {
        return super.getDes();
    }


    @Override
    protected void print() {
        System.out.println(getName());
    }
}
```

### 外观模式

通过定义一个一致的接口，用以屏蔽内部子系统的细节，使得调用端只跟接口发生调用，不需关心内部细节。

外观类(facade)为调用端提供统一的调用接口，外观类知道哪些子系统负责负责处理请求，从而将调用端的请求代理给适当的子系统对象。调用者(Client)外观接口的调用者。子系统的集合指模块或者子系统，处理外观类对象指派的任务。

外观模式在MyBatis框架中的`Configuration.class`的`newMetaObject()`有应用。`Configuration.class`就是一个外观类。下面有很多Factory作为子系统。

**优点**

1. 通过合理的使用外观模式，更好地划分访问层次
2. 屏蔽了子系统细节，降低了客户端对子系统使用的复杂性

```java
public class Client {
	public static void main(String[] args) {
        HomeTheaterFacade homeTheaterFacade = new HomeTheaterFacade(); 
        homeTheaterFacade.ready();
        homeTheaterFacade.play();
        homeTheaterFacade.end();
    }
}

public class HomeTheaterFacade {
    //定义各个子系统对象
    private TheaterLight theaterLight; 
    private Popcorn popcorn;
    private Stereo stereo; 
    private Projector projector; 
    private Screen screen;
    private DVDPlayer dVDPlayer;

    //构造器
    public HomeTheaterFacade() {    
        this.theaterLight = TheaterLight.getInstance(); 
        this.popcorn = Popcorn.getInstance(); 
        this.stereo = Stereo.getInstance(); 
        this.projector = Projector.getInstance();
		this.screen = Screen.getInstance(); 
        this.dVDPlayer = DVDPlayer.getInstanc();
    }

    //操作分成 4  步
    public void ready() { 
        popcorn.on(); 
        popcorn.pop(); 
        screen.down(); 
        projector.on(); 
        stereo.on(); 
        dVDPlayer.on(); 
        theaterLight.dim();
    }

    public void play() { 
        dVDPlayer.play();
    }


    public void pause() { 
        dVDPlayer.pause();
    }

    public void end() { 
        popcorn.off();
    	theaterLight.bright(); 
        screen.up(); 
        projector.off(); 
        stereo.off(); 
        dVDPlayer.off();
    }
}

public class Popcorn {
	private static Popcorn instance = new Popcorn();
    
	public static Popcorn getInstance() { 
        return instance;
	}

    public void on() {
    	System.out.println(" popcorn on ");
    }

    public void off() { 
        System.out.println(" popcorn ff ");
    }

    public void pop() {
    	System.out.println(" popcorn is poping	");
    }
}

public class DVDPlayer {

	//使用单例模式, 使用饿汉式
	private static DVDPlayer instance = new DVDPlayer();


    public static DVDPlayer getInstanc() { 
        return instance;
    }

    public void on() { 
        System.out.println(" dvd on ");
    }
    
    public void off() { 
        System.out.println(" dvd off ");
    }

    public void play() {
    	System.out.println(" dvd is playing ");
    }

    public void pause() { 
        System.out.println(" dvd pause ..");
    }
}

public class Projector {

    private static Projector instance = new Projector();

    public static Projector getInstance() { 
        return instance;
    }

    public void on() {
    	System.out.println(" Projector on ");
    }

    public void off() {
    	System.out.println(" Projector ff ");
    }

    public void focus() {
    	System.out.println(" Projector is Projector	");
    }
}

public class Screen {
    private static Screen instance = new Screen();

    public static Screen getInstance() { 
        return instance;
    }

    public void up() { 
        System.out.println(" Screen up ");
    }

    public void down() { 
        System.out.println(" Screen down ");
    }
}


public class Stereo {
    private static Stereo instance = new Stereo();

    public static Stereo getInstance() { 
        return instance;
    }

    public void on() { 
        System.out.println(" Stereo on ");
    }

    public void off() { 
        System.out.println(" Screen off ");
    }

    public void up() {
    	System.out.println(" Screen up.. ");
    }
}

public class TheaterLight {
    private static TheaterLight instance = new TheaterLight();

    public static TheaterLight getInstance() { 
        return instance;
    }

    public void on() {
    	System.out.println(" TheaterLight on ");
    }

    public void off() {
    	System.out.println(" TheaterLight off ");
    }

    public void dim() {
    	System.out.println(" TheaterLight dim.. ");
    }
    
    public void bright() {
    System.out.println(" TheaterLight bright.. ");
    }
}

```

### 享元模式 Flyweight Pattern

运用共享技术有效的支持大量细粒度的对象。通过解决重复对象的内存浪费的问题，提高系统的性能。经典应用场景就是池技术。

<img src="./pics/07design_flyweight.jpg">

Flyweight是抽象的享元角色。他是产品的抽象类, 同时定义出对象的外部状态和内部状态的接口或实现。

ConcreteFlyWeight是具体的享元角色，是具体的产品类，实现抽象角色定义相关业务。

UnSharedConcreteFlyWeight是不可共享的角色，一般不会出现在享元工厂。

FlyWeightFactory享元工厂类，用于构建一个池容器(集合)， 同时提供从池中获取对象方法。

1)    内部状态指对象共享出来的信息，存储在享元对象内部且不会随环境的改变而改变

2)    外部状态指对象得以依赖的一个标记，是随环境改变而改变的、不可共享的状态。

享元模式在Integer中有使用。如果新创建的Integer对象值在-128~127中，使用享元模式返回。否则，新创建对象返回。

**优点**

1. 降低程序内存占用，减少对象创建，提高效率

**缺点**

1. 需要划分内部和外部状态，还要建立一个工厂

```java
public class Client {
    public static void main(String[] args) {
        // 创建一个工厂类
        WebSiteFactory factory = new WebSiteFactory();

        // 客户要一个以新闻形式发布的网站
        WebSite webSite1 = factory.getWebSiteCategory("新闻");
        webSite1.use(new User("tom"));

        // 客户要一个以博客形式发布的网站
        WebSite webSite2 = factory.getWebSiteCategory("博客");
        webSite2.use(new User("jack"));

        // 客户要一个以博客形式发布的网站
        WebSite webSite3 = factory.getWebSiteCategory("博客");
        webSite3.use(new User("smith"));

        // 客户要一个以博客形式发布的网站
        WebSite webSite4 = factory.getWebSiteCategory("博客");
        webSite4.use(new User("king"));

        System.out.println("网站的分类共=" + factory.getWebSiteCount());
    }
}

//具体网站
public class ConcreteWebSite extends WebSite {
	//共享的部分，内部状态
	private String type = ""; //网站发布的形式(类型)

    //构造器
    public ConcreteWebSite(String type) {
    	this.type = type;
    }

    @Override
    public void use(User user) {
    	System.out.println("网站的发布形式为:" + type + "。使用者是" + user.getName());
    }
}

public abstract class WebSite {
	public abstract void use(User user); //抽象方法
}

// 网站工厂类，根据需要返回压一个网站
public class WebSiteFactory {
	//集合， 充当池的作用
	private HashMap<String, ConcreteWebSite> pool = new HashMap<>();

    //根据网站的类型，返回一个网站, 如果没有就创建一个网站，并放入到池中,并返回
    public WebSite getWebSiteCategory(String type) { 
        if(!pool.containsKey(type)) {
    		//就创建一个网站，并放入到池中
    		pool.put(type, new ConcreteWebSite(type));
    	}
    	return (WebSite)pool.get(type);
    }

	//获取网站分类的总数 (池中有多少个网站类型) 
    public int getWebSiteCount() {
    	return pool.size();
    }
}
```

### 代理模式 Proxy

为对象提供一个替身，以控制对这个对象的访问。增强额外的功能操作，即扩展目标对象的功能。主要是静态代理、动态代理和cglib代理。

#### 静态代理

需要定义接口或者父类，被代理对象(即目标对象)与代理对象一起实现相同的接口或者是继承相同父类。

**优点**

1. 在不修改目标对象的功能前提下, 能通过代理对象对目标功能扩展

**缺点**

1. 因为代理对象需要与目标对象实现一样的接口,所以会有很多代理类
2. 一旦接口增加方法,目标对象与代理对象都要维护

<img src="./pics/08design_staticproxy.jpg">

```java
public class Client {
    public static void main(String[] args) {
        //创建目标对象(被代理对象)
        TeacherDao teacherDao = new TeacherDao();

        //创建代理对象, 同时将被代理对象传递给代理对象
        TeacherDaoProxy teacherDaoProxy = new TeacherDaoProxy(teacherDao);

        //通过代理对象，调用到被代理对象的方法
        //即：执行的是代理对象的方法，代理对象再去调用目标对象的方法
        teacherDaoProxy.teach();
    }
}

//接口
public interface ITeacherDao {
	void teach(); // 授课的方法
}

public class TeacherDao implements ITeacherDao {
    @Override
    public void teach() {
        System.out.println(" 老师授课中	。。。。。");
    }
}

//代理对象,静态代理
public class TeacherDaoProxy implements ITeacherDao{
	private ITeacherDao target; // 目标对象，通过接口来聚合

    //构造器
    public TeacherDaoProxy(ITeacherDao target) { 
        this.target = target;
    }

    @Override
    public void teach() {
        System.out.println("开始代理.."); //方法
        target.teach();
        System.out.println("提交。。。。。"); //方法
    }
}
```

#### 动态代理

也称为JDK 代理、接口代理。代理对象,不需要实现接口，但是目标对象要实现接口，否则不能用动态代理。代理对象的生成，是利用 JDK 的 API，动态的在内存中构建代理对象。

**JDK 中生成代理对象的 API**

1)   代理类所在包: java.lang.reflect.Proxy

2)   JDK 实现代理只需要使用 **newProxyInstance** 方法,但是该方法需要接收三个参数,完整的写法是:

```java
static Object newProxyInstance(ClassLoader loader, Class<?>[] interfaces,InvocationHandler h)
```

<img src="./pics/09design_dynamic_proxy.jpg">

```java
public class Client {

	public static void main(String[] args) {
        //创建目标对象
        ITeacherDao target = new TeacherDao();

        //给目标对象，创建代理对象, 可以转成 ITeacherDao
        ITeacherDao proxyInstance = (ITeacherDao)new ProxyFactory(target).getProxyInstance();

        // proxyInstance=class com.sun.proxy.$Proxy0内存中动态生成了代理对象
        System.out.println("proxyInstance=" + proxyInstance.getClass());

        //通过代理对象，调用目标对象的方法
        proxyInstance.teach();
        proxyInstance.sayHello(" tom ");
    }

}

//接口
public interface ITeacherDao {
    void teach(); // 授课方法
    void sayHello(String name);
}

public class ProxyFactory {

	//维护一个目标对象
    private Object target;

    public ProxyFactory(Object target) {
    	this.target = target;
	}

    //给目标对象  生成一个代理对象
    public Object getProxyInstance() {
/**
newProxyInstance(ClassLoader loader, Class<?>[] interfaces, InvocationHandler h)
1. ClassLoader loader: 指定当前目标对象使用的类加载器, 获取加载器的方法固定
2. Class<?>[] interfaces: 目标对象实现的接口类型，使用泛型方法确认类型
3. InvocationHandler h: 事情处理，执行目标对象的方法时，会触发事情处理器方法, 会把当前执行的目标对象方法作为参数传入
*/
    return Proxy.newProxyInstance(target.getClass().getClassLoader(), target.getClass().getInterfaces(),
        new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                System.out.println("JDK 代理开始~~");
                //反射机制调用目标对象的方法
                Object returnVal = method.invoke(target, args);
                System.out.println("JDK 代理提交"); 
                return returnVal;
            }
        });
	}
}

public class TeacherDao implements ITeacherDao {
    @Override
    public void teach() {
        System.out.println(" 老师授课中.... ");
    }

    @Override
    public void sayHello(String name) {
        System.out.println("hello " + name);
    }
}
```

#### Cglib代理

有时候目标对象只是一个单独的对象，并没有实现任何的接口，这个时候可使用目标对象子类来实现代理。也叫作子类代理，它是在内存中构建一个子类对象从而实现对目标对象功能扩展，有些书也将Cglib 代理归属到动态代理。Cglib 包的底层是通过使用字节码处理框架 ASM 来转换字节码并生成新的类。

**实现步骤**

1. 需要引入 cglib 的 jar 文件 
2. 在内存中动态构建子类，注意代理的类不能为 final，否则报错`java.lang.IllegalArgumentException`

3. 目标对象的方法如果为 final/static，那么就不会被拦截，即不会执行目标对象额外的业务方法

<img src="./pics/10design_cglib.jpg">

```java
public class TeacherDao {
    public String teach() {
        System.out.println("老师授课中，cglib代理不需要实现接口"); 
        return "hello";
    }
}

public class ProxyFactory implements MethodInterceptor {

    private Object target;
    public ProxyFactory(Object target) {
    	this.target = target;
    }

    public Object getProxyInstance() {
        //1. 创建一个工具类
        Enhancer enhancer = new Enhancer();
        //2. 设置父类
        enhancer.setSuperclass(target.getClass());
        //3. 设置回调函数
        enhancer.setCallback(this);
        //4. 创建子类对象，即代理对象
        return enhancer.create();
    }

    //重写intercept 方法，会调用目标对象的方法
    @Override
    public Object intercept(Object arg0, Method method, Object[] args, MethodProxy arg3) throws Throwable {
    	System.out.println("Cglib 代理模式 ~~ 开始"); 
        Object returnVal = method.invoke(target, args); 		
        System.out.println("Cglib 代理模式 ~~ 提交"); 
        return returnVal;
    }
}

public class Client {
    public static void main(String[] args) {
        //创建目标对象
        TeacherDao target = new TeacherDao();
        //获取到代理对象，并且将目标对象传递给代理对象
        TeacherDao proxyInstance = (TeacherDao) new ProxyFactory(target).getProxyInstance();

        //执行代理对象的方法，触发intecept方法，从而实现对目标对象的调用
        String res = proxyInstance.teach(); 	
        System.out.println("res=" + res);
    }
}
```

#### 变体

防火墙代理，缓存代理，远程代理，同步代理。



