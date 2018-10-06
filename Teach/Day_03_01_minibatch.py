# Day_03_01_minibatch.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01

    for _ in range(m):
        z = np.dot(x, w)        # (100, 1) = (100, 3) @ (3, 1)
        h = sigmoid(z)          # (100, 1)
        e = h - y               # (100, 1) = (100, 1) - (100, 1)
        g = np.dot(x.T, e)      # (3, 1) = (3, 100) @ (100, 1)
        w -= lr * g             # (3, 1) -= scalar * (3, 1)

    return w.reshape(-1)


def gradient_stochastic_1(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = i % m
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3,) = (3,) * scalar
        w -= lr * g             # (3,) -= scalar * (3,)

    return w


def gradient_stochastic_2(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for _ in range(m * 10):
        p = np.random.randint(m)
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3,) = (3,) * scalar
        w -= lr * g             # (3,) -= scalar * (3,)

    return w


def gradient_minibatch(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    for _ in range(epochs):
        for j in range(iteration):
            s = j * batch_size
            f = s + batch_size

            z = np.dot(x[s:f], w)   # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[s:f]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[s:f].T, e) # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g             # (3, 1) -= scalar * (3, 1)

    return w.reshape(-1)


def gradient_minibatch_random_1(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    for _ in range(epochs):
        for j in range(iteration):
            idx = np.random.choice(range(m), batch_size, replace=False)

            z = np.dot(x[idx], w)   # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[idx]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[idx].T, e) # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g             # (3, 1) -= scalar * (3, 1)

    return w.reshape(-1)


def gradient_minibatch_random_2(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    orders = np.arange(m)

    for _ in range(epochs):
        for j in range(iteration):
            s = j * batch_size
            f = s + batch_size
            idx = orders[s:f]

            z = np.dot(x[idx], w)   # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[idx]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[idx].T, e) # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g             # (3, 1) -= scalar * (3, 1)

        np.random.shuffle(orders)

    return w.reshape(-1)


# bad.
def gradient_minibatch_random_3(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iteration = m // batch_size

    for i in range(epochs):
        for j in range(iteration):
            s = j * batch_size
            f = s + batch_size
            idx = range(s, f)

            z = np.dot(x[idx], w)   # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[idx]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[idx].T, e) # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g             # (3, 1) -= scalar * (3, 1)

        # np.random.seed(i)
        # np.random.shuffle(x)
        # np.random.seed(i)
        # np.random.shuffle(y)

        seed = np.random.randint(100000)
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)

    return w.reshape(-1)


def decision_boundary(w, c):
    b, w1, w2 = w
    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 *  4 + b) / w2

    plt.plot([-4, 4], [y1, y2], c)

    # y = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x1 + w2 * x2 + b
    # -(w1 * x1 + b) = w2 * x2
    # -(w1 * x1 + b) / w2 = x2


action = np.loadtxt('Data/action.txt')
action = preprocessing.add_dummy_feature(action)
# print(action[:3])

xx = action[:, :-1]
yy = action[:, -1:]
# print(xx.shape, yy.shape)       # (100, 3) (100, 1)

for _, x1, x2, y in action:
    # print(x1, x2)
    plt.plot(x1, x2, 'ro' if y else 'go')

decision_boundary(gradient_descent(xx, yy), 'r')
# decision_boundary(gradient_stochastic_1(xx, yy), 'g')
# decision_boundary(gradient_stochastic_2(xx, yy), 'b')
# decision_boundary(gradient_minibatch(xx, yy), 'y')
# decision_boundary(gradient_minibatch_random_1(xx, yy), 'k')
# decision_boundary(gradient_minibatch_random_2(xx, yy), 'c')
decision_boundary(gradient_minibatch_random_3(xx, yy), 'm')

plt.show()


