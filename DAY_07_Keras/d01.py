import numpy as np

class ReLu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = x>0
        out=np.where(self.mask>0,x, 0)
        return out
    def backward(self, dout):
        dx = np.where(self.mask, dout,0)
        return dx

# x = np.array([1.0, -0.5])
x = 1.0
relu = ReLu()
print(relu.forward(x))
print(relu.backward(dout=2))
