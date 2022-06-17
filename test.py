'''
@Author: Dong Yi

@Description: 尝试一下多个线程里面 能不能再开多线程分批处理数据
'''
import numpy as np

# a = np.array([1,2,3,4])
# b = np.array([1,0,0,1])
# c = 2
#
# score = b[a < c]
# print(score)
# print(a < c)

# a = [[1,2,3],[2,3,4]]
#
# ave = np.mean(a)
#
# print(ave)

from multiprocessing import Pool, cpu_count
import os
import time
from multiprocessing import Process

from MyThread import MyThread


def sub_task(n):
    x = []
    y = []
    for i in range(100):
        x.append(i)
        y.append(i*2)
    return x,y


def task():
    print('%s 子进程 is run....' %os.getpid())
    print("%s"%os.getpid()+"开始分线程执行子任务...")
    pool = Pool(processes=3)
    result1 = pool.apply_async(sub_task, args=(1,))
    result2 = pool.apply_async(sub_task, args=(2,))
    result3 = pool.apply_async(sub_task, args=(3,))

    # p1 = Process(target=sub_task, args=(1,))
    # p2 = Process(target=sub_task, args=(2,))
    # p1.start()
    # p2.start()

    pool.close()
    pool.join()

    a, b = result1.get()
    print(result1.get())

    # p1.join()
    # p2.join()


    time.sleep(1)


if __name__=='__main__':
    a =[3,4]
    b = a.copy()
    b.remove(3)
    print(a,b)


    # a = np.zeros(5)
    # a[3] = 1
    # a[3] = 2
    # print(a)

    # a = {'John': 60, 'Alice': 95, 'Paul': 80, 'James': 75, 'Bob': 85}
    #
    # name = list(a.keys())[list(a.values()).index(75)]
    #
    # print(name)



    # print('%s 主进程 is run....' %os.getpid())
    # thread1 = MyThread(task)
    # thread2 = MyThread(task)
    #
    # thread1.start()
    # thread2.start()
    #
    # thread1.join()
    # thread2.join()
    #
    # print("主进程结束")
