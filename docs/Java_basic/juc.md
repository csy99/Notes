### 内存可见性

JVM为每一个线程分配一个独立的缓存，以提高效率。

内存可见性问题：两个线程对于共享数据的操作，彼此不可见。

下面这段代码主线程的循环会一直执行。因为根本没有重新从堆内存获取flag的取值。

```java
public class TestVolatile {
    public static void main(String[] args) {
        ThreadDemo td = new ThreadDemo();
        new Thread(td).start();
        
        while(true) {
            if (td.isFlag()) {
                System.out.println("Succeed");
                break;
            }
        }
    }
}

class ThreadDemo implements Runnable {
    private boolean flag = false;
    
    @override
    public void run() {
        try {
            Thread.sleep(200);
        } catch (Exception e){}
        
        flag = true;
        System.out.println("flag is true");
    }
    
    public boolean isFlag() {
        return flag;
    }
    
    public void setFlag(boolean f) {
        flag = f;
    }
    
    public boolean getFlag() {
        return flag;
    }
}
```

一个解决办法是加入同步锁synchronized。效率较低。

```java
while(true) {
    synchronized(td) {
        if (td.isFlag()) {
            System.out.println("Succeed");
            break;
        }
    }
}
```

#### volatile关键字

第二个解决办法是加volatile关键字。当多个线程进行操作共享数据时，可以保证内存中的数据可见。取消了代码的重排序。相较于synchronized是一种轻量级的同步策略。但是volatile不具备互斥性，而且不能保证变量的原子性。

```java
private volatile boolean flag = false;
```

### 原子性和CAS算法

i++的原子性问题：实际上分为3个步骤，读-改-写。

```java
int i = 10;
i = i++;  // 10
```

在底层实际上是如下操作

```java
int tmp = i;
i = i + 1;
i = tmp;
```

如下代码会产生原子性问题。

```java
public class TestAtomic {
    public static void main(String[] args) {
        AtmoicDemo ad = new AtomicDemo();
        for (int i = 0; i < 10; i++) {
            new Thread(ad).start();
        }
    }
}

class AtomicDemo implements Runnable {
    private int serialNo = 0;
    
    public void run() {
        Thread.sleep(200);
        System.out.println(Thread.currentThread().getName() + ":" + get());
    }
    
    public int get() {
        return serialNo++;
    }
}
```

#### 原子变量

jdk1.5后`java.util.concurrent.atomic`包提供了常用的原子变量。使用volatile保证内存可见性，和CAS算法保证数据原子性。

```java
private AtomicInteger serialNo = 0;

public int get() {
    return serialNo.getAndIncrement();
}
```

#### CAS算法

compare and swap。是硬件对于并发操作共享数据的支持。包含内存值v，预估值a，更新值b。

如下代码进行了简单的模拟。

```java
public class TestAtomic {
    public static void main(String[] args) {
        AtmoicDemo ad = new AtomicDemo();
        for (int i = 0; i < 10; i++) {
            new Thread(new Runnable() {
                public void run() {
                    int expected = cas.get();
                    boolean b = cas.compareAndSet(expected, randomVal);
                }
            }).start();
        }
    }
}

class CompareAndSwap {
    private int val;
    
    public synchronized int get() {
        return val;
    }
    
    public synchronized int compareAndSwap(int expected, int updated) {
    	int old = val;
        if (old == expected)
            val = updated;
        return old;
    }
    
    public synchronized boolean compareAndSet(int expected, int updated) {
        return expected == compareAndSwap(expected, updated);
    }
}
```

### 同步容器 

Concurrent HashMap采用锁分段机制concurrent level。默认有16个段(segment)。每个segment初始是size为16的哈希表，装着链表。与HashTable的区别实质上就是并行与串行的区别，极大地提升了效率。

jdk1.8以后把Concurrent HashMap也换成了CAS算法。

其他的容器有CopyOnWriteArrayList，CopyOnWriteSet等。

```java
public class TestThread {
    public static void main(String[] args) {
        HelloThread ht = new HelloThread();
        for (int i = 0; i < 10; i++)
            new Thread(ht).start();
    }
}

class HelloThread implements Runnable {
    // 用这个如下代码会报错并发修改异常。
    private static List<String> list = Collections.synchronizedList(new ArrayList<String>());
    // 不会出现错误
    private static CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList();
    
    static {
        list.add("AA");
        list.add("BB");
        list.add("CC");
    }
    public void run() {
        Iterator<String> it = list.iterator();
        while (it.hashNext()) {
            System.out.println(it.next());
            list.add("aa");
        }
    }
}
```

### 闭锁 CountDownLatch

是一个同步辅助类。在完成一组正在其他线程中执行的操作之前，它允许一个或多个线程一直等待。在完成某些运算时，只有其他所有线程的运算全部完成，当前运算才继续执行。

如下代码，我们希望main线程最后执行，以达到计算时间的目的。需要用到闭锁。

```java
public class TestLatch {
    public static void main(String[] args) {
        final CountDownLatch lat = new CountDownLatch(10);
        LatchDemo ld = new LatchDemo(lat);
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10; i++)
            new Thread(ld).start();
        try {
        	latch.await();
        } catch(InterruptedException e) {}
        long end = System.currentTimeMillis();
        System.out.println("time is:" + (end - start));
    }
}

class LatchDemo implements Runnable {
   	private CountDownLatch latch;
    
    public LatchDemo(CountDownLatch lat) {
        latch = lat;
    }
    
    public void run() {
        synchronized (this) {
            try {
                for (int i = 0; i < 5000; i++)
                    if (i % 2 == 0)
                        System.out.println(i);
            } finally {
                latch.countDown();  // 不为0，main线程无法执行
            }    
        }
    }
}
```

### 实现Callable接口

创建执行线程的方式三。相较于实现Runnable接口的方式，方法有返回值，并且可以抛出异常。执行callable需要FutureTask实现类的支持，接受运算结果。FutureTask也可以用于闭锁操作。

```java
public class TestCallable {
    ThreadDemo td = new ThreadDemo();
    FutureTask<Integer> res = new FutureTask(td);
    new Thread(res).start();
    //接受结果
    try {
    	System.out.println(res.get());
    } catch (Exception e) {}
}

class ThreadDemo implements Callable<Integer> {
   	private CountDownLatch latch;
    
    public Integer call() throws Exception{
		int sum = 0;
        for (int i = 0; i < 100; i++)
            sum += i;
        return sum;
    }
}
```

### 同步锁 Lock

用于解决多线程安全问题的方式：同步代码块，同步方法，同步锁（显式锁）。

```java
public class TestLock {
    public static void main() {
        Ticket ticket = new Ticket();
        new Thread(ticket, "1号窗口").start();
        new Thread(ticket, "2号窗口").start();
        new Thread(ticket, "3号窗口").start();
    }
}

class Ticket implements Runnable {
    private int tick = 100;
    private Lock lock = new ReentrantLock();
    
    public void run() {
        lock.lock();
        try{
            while (tick > 0) {
                String name = Thread.currentThread.getName();
                System.out.println(name + "完成售票，余票：" + --tick);
            }
        } finally {
            lock.unlock();
        }
        
    }
}
```

### 等待唤醒机制

生产者和消费者案例。如下代码有可能产生虚假唤醒。主程序没办法结束。解决方法是把else去掉。但是如果有多个生产者和多个消费者，仍然会存在问题。原因是wait()应该使用在循环中。

```java
public class TestPC {
    psvm {
        Clerk clk = new Clerk();
        Productor pro = new Productor(clk);
        Consumer cus = new Consumer(clk);
        new Thread(pro, "生产者A").start();
        new Thread(cus, "消费者B").start();
    }
}

class Clerk {
    private int product = 0;
    
    public synchronized void get() {
        if (product >= 1) {
            System.out.println("无法添加");
            try {
                this.wait();
            } catch(Exception e) {}
        } else {
            System.out.println(Thread.currentThread().getName() + ++product);
            this.notifyAll();
        }
    }
    
    public synchronized void sell() {
        if (product <= 0) {
            System.out.println("无法卖货");
            try {
                this.wait();
            } catch(Exception e) {}
        } else {
            System.out.println(Thread.currentThread().getName() + --product);
            this.notifyAll();
        }
    }
}

class Productor implements Runnable{
    private Clerk clerk;
    
    public Productor(Clerk clerk) {
        this.clerk = clerk;
    }
    
    public void run() {
        for (int i = 0; i < 20; i++) {
            Thread.sleep(200);
            clerk.get();
        }
    }
}

class Consumer implements Runnable {
    private Clerk clerk;
    
    public Productor(Clerk clerk) {
        this.clerk = clerk;
    }
    
    public void run() {
        for (int i = 0; i < 20; i++)
            clerk.sell();
    }    
}
```

### Condition线程通信

```java
public class TestPC {
    psvm {
        Clerk clk = new Clerk();
        Productor pro = new Productor(clk);
        Consumer cus = new Consumer(clk);
        new Thread(pro, "生产者A").start();
        new Thread(cus, "消费者B").start();
    }
}

class Clerk {
    private int product = 0;
    private Lock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();
    
    public void get() {
        lock.lock();
        try{
            if (product >= 1) {
                System.out.println("无法添加");
                try {
                    condition.await();
                } catch(Exception e) {}
            } else {
                System.out.println(Thread.currentThread().getName() + ++product);
                condition.signalAll();
            }
        } finally {
            lock.unlock();
        }
    }
    
    public void sell() {
        lock.lock();
        try{
            if (product <= 0) {
                System.out.println("无法卖货");
                try {
                    condition.await();
                } catch(Exception e) {}
            } else {
                System.out.println(Thread.currentThread().getName() + --product);
                condition.signalAll();
            }
        } finally {
            lock.unlock();
        }
    }
}

class Productor implements Runnable{
    private Clerk clerk;
    
    public Productor(Clerk clerk) {
        this.clerk = clerk;
    }
    
    public void run() {
        for (int i = 0; i < 20; i++) {
            Thread.sleep(200);
            clerk.get();
        }
    }
}

class Consumer implements Runnable {
    private Clerk clerk;
    
    public Productor(Clerk clerk) {
        this.clerk = clerk;
    }
    
    public void run() {
        for (int i = 0; i < 20; i++)
            clerk.sell();
    }    
}
```

### 线程按序交替

```java
public class TestAlternate {
    psvm() {
        AlternateDemo ad = new AlternateDemo();
        new Thread(new Runnable () {
            public void run() {
                for (int i = 1; i < 20; i++)
                    ad.loopA(i).start();
            }
        });
        new Thread(new Runnable () {
            public void run() {
                for (int i = 1; i < 20; i++)
                    ad.loopB(i).start();
            }
        });
        new Thread(new Runnable () {
            public void run() {
                for (int i = 1; i < 20; i++)
                    ad.loopC(i).start();
            }
        });
    }
}

class AlternateDemo {
    int tid = 1; 
    private Lock lock = new ReentrantLock();
    private Condition cond1 = lock.newCondition();
    private Condition cond2 = lock.newCondition();
    private Condition cond3 = lock.newCondition();
    
    public void loopA(int loop) {
        lock.lock();
        try{
            if (tid != 1)
                cond1.await();
            for (int i = 0; i < 5; i++)
                System.out.println(i + "A" + "\t" + loop);
            tid = 2;
            cond2.signal();
        } finally {
            lock.unlock();
        }
    }
    
    public void loopB(int loop) {
        lock.lock();
        try{
            if (tid != 2)
                cond2.await();
            for (int i = 0; i < 10; i++)
                System.out.println(i + "B" + "\t" + loop);
            tid = 3;
            cond3.signal();
        } finally {
            lock.unlock();
        }
    }
    
    public void loopC(int loop) {
        lock.lock();
        try{
            if (tid != 3)
                cond3.await();
            for (int i = 0; i < 20; i++)
                System.out.println(i + "C" + "\t" + loop);
            tid = 1;
            cond1.signal();
        } finally {
            lock.unlock();
        }
    }    
}
```

### 读写锁ReadWriteLock

```java
class RWDenmo {
    private int num = 0;
    private ReadWriteLock lock = new ReentrantReadWriteLock();
    
    public void read() {
        lock.readLock().lock();
        try {
            System.out.println(Thread.currentThread().getName() + ":" + num);
        } finally {
            lock.readLock().unlock();
        }
        
    }
    
    public void write(int n) {
        lock.writeLock().lock();
        try {
            System.out.println(Thread.currentThread().getName();
            num = n;
        } finally {
            lock.writeLock().unlock();
        }
    }
}
```

### 线程八锁

关键点在于非静态方法的锁默认为this，静态方法的锁是对应的Class实例。在某一个时刻内，只能有一个线程持有锁，无论几个方法。

1. 两个普通同步方法，两个线程，标准打印 //1,2,3
2. 给get1()新增sleep(200)，打印 //3,1,2（因为get3()不是synchronized）
3. 将get1()改成静态同步方法，打印 //2,1（暂时不考虑3）
4. 将get1()和get()2都改成静态同步方法，打印 //1,2（暂时不考虑3）
5. 将get1()改成静态同步方法，使用num1调用，将get2()使用num2调用（两个Number对象）打印 //2,1
6. 将get1()和get()2都改成静态同步方法，分别使用num1、num2调用（两个Number对象）打印 //1,2

```java
public class Test8Monitor {
    psvm() {
        Number num = new Number();
        new Thread(new Runnable(){
            public void run() {
                number.get1();
            }
        }).start();
        new Thread(new Runnable(){
            public void run() {
                number.get2();
            }
        }).start();        
        new Thread(new Runnable(){
            public void run() {
                number.get2();
            }
        }).start();  
    }
}

class Number {
    public synchronized void get1() {
        System.out.println(1);
    }
    
    public synchronized void get2() {
        System.out.println(2);
    }
    
    public void get3() {
        System.out.println(3);
    }    
}
```

### 线程池

频繁地创建和销毁线程很浪费资源。解决办法是提供了一个线程队列，队列中保存着所有等待状态的线程。避免了创建销毁的额外开销。`java.util.concurrent.Executor`负责线程的使用和调度的根接口，`ThreadPoolExecutor`是线程池实现类，`ScheduledExecutorService`是负责线程调度的子接口，`ScheduledThreadPoolExecutor`继承了`ThreadPoolExecutor`而且实现了`ScheduledExecutorService`。

`ExecutorService newFixedThreadPool()`创建固定大小的线程池。`ExecutorService newCachedThreadPool()`是缓存线程池，线程数量不固定，可以根据需求自动更改。`ExecutorService newSingleThreadPool()`创建一个线程池，池中只有一个线程。

`ScheduledExecutorService newScheduledThreadPool()`创建固定大小的线程池，可以延迟或定时的执行任务。

```java
public class TestPool {
    psvm() {
        ThreadPoolDemo tpd = new ThreadPoolDemo();
        //new Thread(tpd).start();
        //new Thread(tpd).start();
        ExecutorService pool = Executors.newFixedThreadPool(5);
        for (int i = 0; i < 10; i++)
        	pool.submit(tpd);  // 分配任务
        pool.shutdown();
    }
}

class ThreadPoolDemo implements Runnable {
    private int i = 0;
    
    public void run() {
        while (i <= 100)
            System.out.println(Thread.currentThread().getName() + ":" + i++);
    }
}
```

可延时的。

```java
public class TestScheduledPool {
    psvm() throws Exception {
        ScheduledExecutorService pool = Executors.newScheduledThreadPool(5);
        Future<Integer> res = pool.schedule(new Callable<Integer>() {
            public Integer call() throws Exception {
                System.out.println(Thread.currentThread().getName()+":");
            }
        }, 3, TimeUnit.MINUTES);
        System.out.println(res.get());
        pool.shutdown();
    }
}
```

### ForkJoinPool分支合并框架

一个大任务，拆分成若干个小任务，再将所有小任务运算的结果进行join汇总。

采用工作窃取模式(work stealing)，。

```java
public class TestForkJoin {
    psvm() {
        ForkJoinPool pool = new ForkJoinPool();
        ForkJoinTask<Long> task = new ForkJoinSum(0, 1_000_000_000L);
        Long sum = pool.invoke(task);
        System.out.println(sum);
    }
}

class ForkJoinSum extends RecursiveTask<Long> {
    private static final long serialUID = -988L;
    private long start;
    private long end;
    private static final long threshold = 10000L;
    
    public ForkJoinSum(long s, long e) {
        start = s;
        end = e;
    }
    
    Long compute() {
        long len = end - start;
        if (len <= threshold) {
            long sum = 0L;
            for (long i = start; i <= end; i++)
                sum += i;
            return sum;
        } else {
            long mid = (start + end) / 2;
            ForkJoinSum left = new ForkJoinSum(start, middle);
            left.fork(); //进行拆分，压入线程队列
            ForkJoinSum right = new ForkJoinSum(middle+1, end);
            right.fork(); //进行拆分，压入线程队列     
            return left.join()+right.join();
        }
    }
}
```











# Reference

- JUC并发编程 by 尚硅谷。