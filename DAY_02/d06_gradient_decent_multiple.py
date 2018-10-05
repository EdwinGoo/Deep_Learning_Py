import matplotlib.pyplot as plt
import numpy as np

def gradient_decent():
    def gradient_decent(x,y,w) :
        c = 0
        for i in range(len(x)) :
            hx = w*x[i]
            c += (hx-y[i])*x[i]
        return c/len(x)

    def show_gradient() :
        x = [1,2,3]
        y = [1,2,3]
        w = 10
        for i in range(1001) :
            g = gradient_decent(x,y,w)
            w -= 0.1 * g
        print(w)

    show_gradient()

# gradient_decent()


def gradient_decent_bias_add():
    def gradient_decent(x, y, w, b) :
        c1, c2 = 0, 0
        for i in range(len(x)) :
            hx = w * x[i] + b
            c1 += (hx-y[i]) * x[i]
            c2 += (hx-y[i])
        return c1/len(x), c2/len(x)

    def show_gradient() :
        x = [1,2,3]
        y = [1,2,3]
        w, b = 10 , 10
        for i in range(1000) :
            g1, g2 = gradient_decent(x, y, w, b)
            w -= 0.1 * g1
            b -= 0.1 * g2

        print(w, b)

    show_gradient()

# gradient_decent_bias_add()

def gradient_decent_multiple():
    def gradient_descent(x, y, w):
        c = [0, 0, 0]
        for i in range(len(x)):
            hx = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2]
            c[0] += (hx - y[i]) * x[i][0]
            c[1] += (hx - y[i]) * x[i][1]
            c[2] += (hx - y[i]) * x[i][2]

        return c[0] / len(x), c[1] / len(x), c[2] / len(x)
    def show_gradient():
        x = [[1., 1., 0.],
             [1., 0., 2.],
             [1., 3., 0.],
             [1., 0., 4.],
             [1., 5., 0.]]
        y = [1, 2, 3, 4, 5]

        w = [10, 10, 10]
        for i in range(1000):
            g1, g2, g3 = gradient_descent(x, y, w)
            w[0] -= 0.1 * g1
            w[1] -= 0.1 * g2
            w[2] -= 0.1 * g3

        print(w)
    show_gradient()

# gradient_decent_multiple()

def gradient_decent_multiple_up():
    def gradient_descent(x, y, w):
        c = np.zeros(3,dtype=np.float32)
        for i in range(len(x)):
            hx = np.sum(w * x[i])
            c += (hx - y[i]) * x[i]
        return c / len(x)

    def show_gradient():
        x = [[1., 1., 0.],
             [1., 0., 2.],
             [1., 3., 0.],
             [1., 0., 4.],
             [1., 5., 0.]]
        # x = np.float32(x)
        y = [1, 2, 3, 4, 5]

        w = np.full(3,fill_value=10,dtype=np.float32)
        for i in range(1000):
            g = gradient_descent(x, y, w)
            w -= 0.1 * g
        print(w)

    show_gradient()

gradient_decent_multiple_up()

