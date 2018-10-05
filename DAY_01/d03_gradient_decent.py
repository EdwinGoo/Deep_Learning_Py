import matplotlib.pyplot as plt

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
    bcost = 1000000
    for i in range(1001) :
        c = cost(x,y,w)
        g = gradient_decent(x,y,w)
        w = w - 0.3 * g
        # if c < 1e-30 : # c == 0은 정말 븅신같은 생각
        if abs(bcost - c) < 1e-30 : # c == 0은 정말 븅신같은 생각
            print(i,c, w)
            break
        bcost = c

    print(w*5)
    print(w*7)

show_gradient()