import numpy as np
import matplotlib.pyplot as plt

def f(x, k=1.0):
    return np.sin(k*x)

def fprime(x, k=1.0):
    return k*np.cos(k*x)

def fprime2(x, k=1.0):
    return -k**2*np.sin(k*x)

def forward_diff(fvals, h):
    """前向差分：一阶精度"""
    return (fvals[1:] - fvals[:-1]) / h

def central_diff(fvals, h):
    """中心差分：二阶精度"""
    return (fvals[2:] - fvals[:-2]) / (2*h)

def second_diff_3point(fvals, h):
    """三点中心差分：二阶精度"""
    return (fvals[:-2] - 2*fvals[1:-1] + fvals[2:]) / (h**2)

def second_diff_5point(fvals, h):
    """五点中心差分：四阶精度"""
    return (-fvals[:-4] + 16*fvals[1:-3] - 30*fvals[2:-2] 
            + 16*fvals[3:-1] - fvals[4:]) / (12*h**2)
