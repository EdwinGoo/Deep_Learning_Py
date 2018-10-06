# Day_01_03_cost_gradient.py
import matplotlib.pyplot as plt


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2

    return c / len(x)


def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) * x[i]

    return c / len(x)


def show_cost():
    # y = ax + b
    # h(x) = wx + b
    # y = 1 * x + 0
    x = [1, 2, 3]
    y = [1, 2, 3]

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        print(w, c)

        plt.plot(w, c, 'ro')
    plt.show()


def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w, old = 10, 100000000
    for i in range(200):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g

        # early stopping.
        # if c < 1e-15:
        if old - c < 1e-15:
            break
        old = c

        # print(i, w)
        print(i, c)

    # 문제
    # w가 1.0이 되도록 수정하세요 (2가지)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print('5 :', w * 5)
    print('7 :', w * 7)


show_cost()
# show_gradient()


# 미분 : 기울기. 순간변화량.
#        x축으로 1만큼 움직였을 때 y축으로 움직인 거리

# y = 3             3=1, 3=2, 3=3
# y = x             1=1, 2=2, 3=3
# y = 2x            2=1, 4=2, 6=3
# y = (x+1)         2=1, 3=2, 4=3
# y = xz

# y = x^2           1=1, 4=2, 9=3 --> 2*x^(2-1) --> 2x
#                                     2*x^(2-1) * x미분
# y = (x+1)^2                     --> 2*(x+1)^(2-1) --> 2(x+1)
#                                 --> 2*(x+1)^(2-1) * (x+1)미분







