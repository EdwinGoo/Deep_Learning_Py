import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

##gradient descent

def cost(x,y,w) :
    c = 0
    for i in range(len(x)) :
        hx = w*x[i]
        c += (hx-y[i])**2
    return c/len(x)

def showcost() :
    x = [1,2,3]
    y = [1,2,3]

    for i in range(-30,50) :
        w = i / 10
        # print(w) 
        c =cost(x,y,w)
        print(w, c) 

        plt.plot(w,c,'bx') ## ro red O bx blue X
    plt.show()

# showcost()

def showtensorflowcost() :
    x = [1,2,3]
    y = [1,2,3]
    
    # 가변값을 placeholder 잡아라 
    w = tf.placeholder(tf.float32)

    hx = w * x
    costEQloss = tf.reduce_mean((hx - y)**2)

    sess = tf.Session()

    for i in range(-30,50) :
        c = sess.run(costEQloss, {w:i/10})
        plt.plot(i/10, c ,'ro')  # x좌표 , y좌표, 표현방식
 
    plt.show()

# showtensorflowcost() 

def showtensorflowcost2() :
    x = [1,2,3]
    y = [1,2,3]
    
    # 가변값을 placeholder 잡아라 
    w = tf.placeholder(tf.float32)

    hx = w * x
    costEQloss = tf.reduce_mean((hx - y)**2)

    ww , cc = [], []

    sess = tf.Session()

    for i in range(-30,50) :
        c = sess.run(costEQloss, {w:i/10})
        ww.append(i/10)
        cc.append(c)

    plt.plot(ww,cc,"ro")
    plt.show()

# showtensorflowcost2() 

def showtensorflowcost3() :
    x = [1,2,3]
    y = [1,2,3]

    ww = np.arange(-3, 5, 0.1).reshape(-1,1)
    cc = np.mean((ww * x - y)**2, axis=1)
    plt.plot(ww,cc,"ro")
    plt.show()

showtensorflowcost3() 