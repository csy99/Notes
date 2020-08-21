# RPC

远程方法调用 Remote Procedure Call

分布式的通信方式。这是一个概念，具体有很多实现。必须要实现序列化。



# 分布式通信

最基本是二进制数据传输TCP/IP。也可以用http或者webservice等其他通讯协议。

e.g.:

User类里面有两个属性，id和name。UserService接口有一个方法是findUserById(). 客户端需要把user id传给服务端，服务端需要传回user对象的信息。

下面代码是最原始的方式，非常不灵活。服务端传输对象需要把对象所有属性传输一遍。

```java
public class Client {
    public static void main() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();        
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(123);
        
        Socket s = new Socket("127.0.0.1", 8888);
        //在java中转成二进制
        s.getOutputStream().write(baos.toByteArray());
        s.getOutputStream().flush();
        
        DataInputStream dis = new DataInputStream(s.getInputStream());
        int id = dis.readInt();
        String name = dis.readUTF();
        User user = new User(id, name);
        dos.close();
        s.close();
    }
}
```

```java
public class Server {
    private static void process(Socket s) throws Exception {
        InputStream in = s.getInputStream();
        OutputStream out = s.getOutputStream();
        DataInputStream dis =new DataInputStream(in);
        DataOutputStream dos = new DataOutputStream(out);
        
        int id = dis.readInt();
        IUserService serv = new UserServiceImpl();
        User user = service.findUserById(id);
        dos.writeInt(user.getId());
        dos.writeUTF(user.getName());
        dos.flush();
    }
}
```

修改一下网络传输的部分，进行封装。在客户端创建一个代理。

```java
public class Client {
    public static void main() {
        Stub stub = new Stub();
        stub.findUserById(123);
    }
}
```

```java
//实际上就是v1版本的client的代码
public class Stub {
    public User findUserById(Integer id) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();        
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(123);
        
        Socket s = new Socket("127.0.0.1", 8888);
        //在java中转成二进制
        s.getOutputStream().write(baos.toByteArray());
        s.getOutputStream().flush();
        
        DataInputStream dis = new DataInputStream(s.getInputStream());
        int id = dis.readInt();
        String name = dis.readUTF();
        User user = new User(id, name);
        dos.close();
        s.close();        
    }
}
```

目前stub只能代理一个方法。无法解决如果接口改变需要大量修改代码的问题。进行迭代，用到代理模式中的动态代理。动态生成`IUserService`接口中的方法的调用。

```java
public class Client {
    public static void main() {
        IUserService service = Stub.getStub();
        service.findUserById(123);
    }
}
```

```java
public class Stub {
    public static IUserService getStub() {
        InvocationHandler h = new InvocationHandler() {
            public Object invoke(Object proxy, Method mtd, Object[] args) throws Throwable {
                Socket s = new Socket("127.0.0.1", 8888);
				
                ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream);
                String methodName = mtd.getName();
                Class[] parametersTypes = mtd.getParameterTypes();
                oos.writeUTF(methodName);
                //方法有可能重载，需要指定传参方法
                oos.writeObject(parametersTypes);
                oos.writeObject(args);
                oos.flush();
                
                DataInputStream dis = new DataInputStream(s.getInputStream());
                int id = dis.readInt();
                String name = dis.readUTF();
                User user = new User(id, name);
                oos.close();
                s.close();        
                return user;
            }
        };
        Object o = Proxy.newProxyInstance(IUserService.class.getClassLoader(), 
                       new Class[]{IUserService.class}, h);
        System.out.println(o.getClass().getName());
        return (IUserService)o;
    }
}
```

```java
public class Server {
    private static void process(Socket s) throws Exception {
        InputStream in = s.getInputStream();
        OutputStream out = s.getOutputStream();
        ObjectInputStream ois =new ObjectInputStream(in);
        DataOutputStream dos = new DataOutputStream(out);  
        
        String methodName = ois.readUTF();
        Class[] parameterTypes = (Class[]) ois.readObject();
        Object[] args = (Object[]) ois.readObject;
        
        IUserService service = new UserServiceImpl();
        Method met = service.getClass().getMethod(methodName, parameterTypes);
        User user = (User)met.invoke(service, args);
        
        dos.writeInt(user.getId());
        dos.writeUTF(user.getName());
        dos.flush();
    }
}
```

目前，服务端返回值仍然是对象拆解之后的属性(ID和Name)。下面这个改进解决了这个问题。

```java
public class Server {
    private static void process(Socket s) throws Exception {
        InputStream in = s.getInputStream();
        OutputStream out = s.getOutputStream();
        ObjectInputStream ois =new ObjectInputStream(in); 
        
        String methodName = ois.readUTF();
        Class[] parameterTypes = (Class[]) ois.readObject();
        Object[] args = (Object[]) ois.readObject;
        
        IUserService service = new UserServiceImpl();
        Method met = service.getClass().getMethod(methodName, parameterTypes);
        User user = (User)met.invoke(service, args);
        
        ObjectOutputStream oos = new ObjectOutputStream(out);
        oos.writeObject(user);
        oos.flush();
    }
}
```

此时，在客户端调用getStub()还只能拿到一个接口中的方法。下面代码把getStub()转换成泛型，解决了这个问题。客户端再从服务注册表找到具体的类。

```java
public class Stub {
    public static Object getStub(Class clazz) {
        InvocationHandler h = new InvocationHandler() {
            public Object invoke(Object proxy, Method mtd, Object[] args) throws Throwable {
                Socket s = new Socket("127.0.0.1", 8888);
				
                ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream);
                String clazzName = clazz.getName();
                String methodName = mtd.getName();
                Class[] parametersTypes = mtd.getParameterTypes();
                oos.writeUTF(clazzName);
                oos.writeUTF(methodName);
                //方法有可能重载，需要指定传参方法
                oos.writeObject(parametersTypes);
                oos.writeObject(args);
                oos.flush();
                
                ObjectInputStream ois = new ObjectInputStream(s.getInputStream());
                Object o = ois.readObject();
                oos.close();
                s.close();        
                return o;
            }
        };
        Object o = Proxy.newProxyInstance(IUserService.class.getClassLoader(), 
                       new Class[]{IUserService.class}, h);
        System.out.println(o.getClass().getName());
        return (IUserService)o;
    }
}
```



# RPC序列化框架

序列化：把对象转化成字节数组。

1. java.io.Serializable
2. Hessian
3. google protobuf
4. facebook Thrift
5. kyro
6. fst
7. json序列化框架(jackson, google Gson, Ali FastJson)
8. xstream

### Hessian

通过Hessian序列化产生的byte数组的长度远小于jdk自带的，去除了无关的信息。时间也要短很多。

```java
public class HelloHessian {
    public static void main() {
        User u = new User(1, "adf");
        byte[] bytes = serialize(u);
        User u1 = (User)deserialize(bytes);
    }
    
    public static byte[] serialize(Object o) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Hessian2Output out = new Hessian2Output(baos);
        out.writeObject(o);
        out.flush();
        byte[] bytes = baos.toByteArray();
        out.close();
        baos.close();
        return bytes;
    }
    
   	public static Object deserialize(byte[] bytes) {
        ByteArrayInputStream bais = new ByteArrayInputStream();
        Hessian2Input input = new Hessian2Input(bais);
        Object o = input.readObject();
        bais.close();
        input.close();
        return o;
    }
}
```





