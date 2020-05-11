sleep、wait和yield都是暂停线程的方法。
# 定义的类
sleep()和yield()方法是定义在Thread类中，而wait()方法是定义在Object类中。
  
# sleep
在Java中Sleep方法有两个，一个只有一个毫秒参数，另一个有毫秒和纳秒两个参数。
```java
sleep(long millis);
sleep(long millis, int nanos);
```
1. 是一个静态方法，让当前执行的线程(而不是调用sleep的线程)sleep指定的时间。  
2. 不会释放锁  
3. 如果其他的线程中断了一个休眠的线程，sleep方法会抛出Interrupted Exception  
4. 休眠的线程在唤醒之后会先进入就绪态runnable，不保证能获取到CPU   
 
# wait
实现线程间通信，应在同步代码块中调用。  
1. 是一个实例方法，并且只能在其他线程调用本实例的notify()方法时被唤醒。  
2. 会释放锁   
3. 通常有条件地执行，线程会一直处于wait状态，直到某个条件变为真  

# yield
释放线程所占有的CPU资源，从而让其他线程有机会运行，但是不能保证某个特定的线程能够获得CPU资源。
谁能获得CPU完全取决于调度器，在有些情况下调用yield方法的线程甚至会再次得到CPU资源。  
1. 让优先级大于等于执行yield的线程的线程有机会执行。 





# Reference
- https://www.jianshu.com/p/25e959037eed  


