
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f2 = lambda x: np.sum(x**2, 0)
x0 = np.arange(-5, 5, 0.25)
x1 = np.arange(-5, 5, 0.25)

X0, X1 = np.meshgrid(x0,x1)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X0, X1, f2(np.array([X0, X1])))
plt.show()

def numerical_gradient(f, x):
    h = 1e-4
    기울기 = np.zeros_like(x)
    
    # 편미분: 각 축별로 미분 수행
    for idx in range(len(x)):
        val = x[idx]
        # f(x+h)
        x[idx] = val + h
        fxh1 = f(x) 
        # f(x-h)
        x[idx] = val -h
        fxh2 = f(x)
        # 수치 미분
        기울기[idx] = (fxh1 - fxh2) / (2*h)
        
    return 기울기

numerical_gradient(f2, np.array([3., 4.]))
numerical_gradient(f2, np.array([0., 4.]))
numerical_gradient(f2, np.array([3., 0.]))