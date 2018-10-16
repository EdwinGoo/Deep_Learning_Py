import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def MCP_N(x,w,b) :
    z = np.dot(x,w) + b
    return z

def activation(z) :
    y = 1 if z > 0 else -1
    return y

def AND(x1, x2) :
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    z = MCP_N(x,w,b)
    y = activation(z)
    y = 0 if y == -1 else y
    return y 

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    # MCP 뉴런
    z = MCP_N(x,w,b)
    y = activation(z)
    y = 0 if y == -1 else y
    return y

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    # MCP 뉴런
    z = MCP_N(x,w,b)
    y = activation(z)
    y = 0 if y == -1 else y
    return y

def test(logic) : 
   for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
        y = logic(x1, x2)
        print(x1, x2, '|', y)

test(AND)