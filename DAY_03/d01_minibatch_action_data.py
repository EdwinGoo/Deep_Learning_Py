import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

action = np.loadtxt("./DATA/action.txt") # [-323, -293]
action = preprocessing.add_dummy_feature(action) #바이어스 추가해줌 [1,-323,-293]
xx = action[:,:-1] # print(xx.shape)
yy = action[:,-1:] # print(yy.shape)

for _, x1, x2, y in action :
    plt.plot(x1,x2, "ro" if y else "bx")
# plt.show()

def sigmoid_my(z) :
    return 1 / (1+np.exp(-z))

def gradient_descent(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros([n,1]) # (3,1)
    lr = 0.01 

    for _ in range(m):
        z = np.dot(x,w) # 100,1
        h = sigmoid_my(z)
        e = h - y
        g = np.dot(x.T, e)
        w -= lr * g

    return w.reshape(-1)

def gradient_stochastic1(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros(n) # 3
    
    # print(w.shape) # (3,)
    # print(w) # [0.,0.,0.]

    lr = 0.01 

    for i in range(m * 10):
        p = i % m 
        z = np.sum(x[p] * w) # scalar sum((3,i) * (3,))
        h = sigmoid_my(z)
        e = h - y[p]
        g = x[p] * e
        w -= lr * g

    return w

def gradient_minibatch(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros([n,1]) # (3,1)
    lr = 0.01 
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    for _ in range(epochs):
        for j in range(iteration) :
            start = j * batch_size
            finish = start + batch_size
            z = np.dot(x[start:finish],w) # 100,1
            h = sigmoid_my(z)
            e = h - y[start:finish]
            g = np.dot(x[start:finish].T, e)
            w -= lr * g

    return w.reshape(-1)    

def gradient_stochastic2(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros(n) # 3
    
    # print(w.shape) # (3,)
    # print(w) # [0.,0.,0.]

    lr = 0.01 

    for _ in range(m * 10):
        p = np.random.randint(m)
        z = np.sum(x[p] * w) # scalar sum((3,i) * (3,))
        h = sigmoid_my(z)
        e = h - y[p]
        g = x[p] * e
        w -= lr * g

    return w   

def gradient_minibatch_random(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros([n,1]) # (3,1)
    lr = 0.01 
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    for _ in range(epochs):
        for j in range(iteration) :
            idx = np.random.choice(range(m),batch_size, replace = False)
            # start = j * batch_size
            # finish = start + batch_size
            z = np.dot(x[idx],w) # 100,1
            h = sigmoid_my(z)
            e = h - y[idx]
            g = np.dot(x[idx].T, e)
            w -= lr * g

    return w.reshape(-1)

def gradient_minibatch_random2(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros([n,1]) # (3,1)
    lr = 0.01 
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    orders = np.arange(m)

    for _ in range(epochs):
        np.random.shuffle(orders)
        for j in range(iteration) :
            start = j * batch_size
            finish = start + batch_size

            idx = orders[start:finish]

            z = np.dot(x[idx],w) # 100,1
            h = sigmoid_my(z)
            e = h - y[idx]
            g = np.dot(x[idx].T, e)
            w -= lr * g

    return w.reshape(-1)


def gradient_minibatch_random2(x,y) :
    m, n = x.shape # (100,3)
    w = np.zeros([n,1]) # (3,1)
    lr = 0.01 
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    orders = np.arange(m)

    for _ in range(epochs):
        np.random.shuffle(orders)
        for j in range(iteration) :
            start = j * batch_size
            finish = start + batch_size

            idx = orders[start:finish]

            z = np.dot(x[idx],w) # 100,1
            h = sigmoid_my(z)
            e = h - y[idx]
            g = np.dot(x[idx].T, e)
            w -= lr * g

    return w.reshape(-1)

def decision_boundary(w,color) :
    b, w1, w2 = w

    y1 = -(w1 * -4 +  b) / w2
    y2 = -(w1 * 4 + b) / w2

    plt.plot([-4,4],[y1,y2],color, linewidth = 3)

decision_boundary(gradient_descent(xx,yy), "r")
decision_boundary(gradient_stochastic1(xx,yy), "g")
decision_boundary(gradient_stochastic2(xx,yy), "b")
decision_boundary(gradient_minibatch(xx,yy), "y")
decision_boundary(gradient_minibatch_random(xx,yy), "k")
decision_boundary(gradient_minibatch_random2(xx,yy), "c")

plt.show()
