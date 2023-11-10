# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:46:43 2023

@author: 14029
"""
import torch
import numpy as np
from torch import sin,exp,cos,log,sqrt
def yi(X):
    return 1+0.25*(X+1)
def ufunc(x,a,k,m):
    #print(len(x.shape))
    if len(x.shape) == 0:
        if x > a:
            return k*(x-a)**m
        if x < -a:
            return k*(-x-a)**m
        else:
            return torch.tensor(0)
    else:
        ten = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if bool(x[i] > a):
                ten[i] = k*(x[i]-a)**m
            if bool(x[i] < -a):
                ten[i] = k*(-x[i]-a)**m
        return ten
    
def sphere(X):
    f1 = torch.sum(X**2,dim=-1)
    return f1

def bbfsphere(X):
    # X is a tensor
    x = x.cpu().data.numpy()
    return torch.from_numpy(X**2).float().to(device)

def schw222(X):
    f2 = torch.sum(torch.abs(X),dim=-1)+torch.prod(torch.abs(X),dim=-1)
    return f2
def schw12(X):
    d = X.shape[-1]
    f3 = 0
    #print(X.shape)
    if len(X.shape) == 2:
        for i in range(d-1):
            f3 = f3 + torch.sum(X[:,:i+1],dim=-1)**2
    else:
        for i in range(d-1):
            f3 = f3 + torch.sum(X[:i+1],dim=-1)**2
    return f3
def schw221(X):
    f4 = torch.max(torch.abs(X),-1).values
    #print(f4)
    return f4
def rose(X):
    d = X.shape[-1]
    rb = 0
    if d != 2:
        if len(X.shape) != 2:
            for i in range(d-1):
                rb = rb + 100*((X[i+1]-X[i]**2)**2)+(1-X[i])**2
        else:
            for i in range(d-1):
                rb = rb + 100*((X[:,i+1]-X[:,i]**2)**2)+(1-X[:,i])**2
    else:
        if len(X.shape) != 2:
            rb = rb + 100*((X[1]-X[0]**2)**2)+(1-X[1])**2
        else:
                rb = rb + 100*((X[:,1]-X[:,0]**2)**2)+(1-X[:,1])**2
    #print(rb)
    return rb
def step(X):
    f6 = torch.sum(torch.floor(X+0.5)**2,dim=-1)
    return f6
def quartic(X):
    d = X.shape[-1]
    f7 = 0
    print(X.shape)
    if len(X.shape) == 2:
        for i in range(d):
            f7 = f7 + i*X[:,i]**4
    else:
        for i in range(d):
            f7 = f7 + i*X[i]**4
    return f7+np.random.rand(X.shape[0])
def schw(X):
    d = X.shape[-1]
    schw = torch.tensor(418.9829*d)-torch.sum((X)*sin(sqrt(torch.abs(X))),dim=-1)
    return schw
def ras(X):
    d = X.shape[-1]
    c = 2*np.pi
    ra = torch.tensor(10*d)+torch.sum((X)**2,dim=-1)-10*torch.sum(cos((X)*c),dim=-1)
    return ra
def ack(X):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = X.shape[-1]
    Ackley1 = -a*exp(-b*sqrt((torch.sum((X)**2,dim=-1))/d))-exp(torch.sum(cos((X)*c),dim=-1)/d)+ a + exp(torch.tensor(1))
    #print(Ackley1)
    return Ackley1
def grie(X):
    s = 1
    d = X.shape[-1]
    for i in range(d):
        if len(X.shape) == 2:
            s = s*cos((X[:,i])/torch.sqrt(torch.tensor(i+1)))
        else:
            s = s*cos((X[i])/torch.sqrt(torch.tensor(i+1)))
    Griewank = torch.sum((X)**2,dim=-1)/4000 - s + torch.tensor(1)
    #print(Griewank)
    return Griewank
# 这里修改删去了.to(device)，测试时可能会出问题
def gpf1(X):
    d = X.shape[-1]
    #print(X.shape)
    if len(X.shape) == 2:
        f12 = np.pi/d * (10*torch.sin(np.pi*yi(X[:,0]))**2+(yi(X[:,d-1])-1)**2)
        for i in range(d):
            f12 = f12 + ufunc(X[:,i],10,100,4)
        for i in range(1,d):
            f12 = f12 + np.pi/d*((yi(X[:,i-1])-1)**2*(1+10*torch.sin(np.pi*yi(X[:,i]))**2))
            
    else:
        f12 = np.pi/d * (10*torch.sin(np.pi*yi(X[0]))**2+(yi(X[d-1])-1)**2)
        for i in range(d):
            f12 = f12 + ufunc(X[i],10,100,4)
        for i in range(1,d):
            f12 = f12 + np.pi/d*((yi(X[i-1])-1)**2*(1+10*torch.sin(np.pi*yi(X[i])))**2)
    return f12
def gpf2(X):
    d = X.shape[-1]
    #print(X.shape)
    if len(X.shape) == 2:
        f13 = 0.1 * (torch.sin(3*np.pi*X[:,0])**2+(X[:,d-1]-1)**2*(1+torch.sin(2*np.pi*X[:,d-1])**2))
        for i in range(d):
            f13 = f13 + ufunc(X[:,i],5,100,4)
        for i in range(1,d):
            f13 = f13 + 0.1*((X[:,i-1]-1)**2*(1+torch.sin(3*np.pi*X[:,i])**2))
            
    else:
        f13 = 0.1 * (torch.sin(3*np.pi*X[0])**2+(X[d-1]-1)**2*(1+torch.sin(2*np.pi*X[d-1])**2))
        for i in range(d):
            f13 = f13 + ufunc(X[i],5,100,4)
        for i in range(1,d):
            f13 = f13 + 0.1*((X[i-1]-1)**2*(1+torch.sin(3*np.pi*X[i])**2))
    return f13
def foxhole(X):
    ass = np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
    f14 = 1/500
    if len(X.shape) == 2:
        for j in range(ass.shape[1]):
            cnt = 0
            for i in range(ass.shape[0]):
                cnt = cnt + (X[:,i]-ass[i][j])**6
            f14 = f14 + 1/(j+cnt)
        return 1/f14
    else:
        for j in range(ass.shape[1]):
            cnt = 0
            for i in range(ass.shape[0]):
                cnt = cnt + (X[i]-ass[i][j])**6
            f14 = f14 + 1/(j+cnt)
        return 1/f14
def kowalik(X):
    a = np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    b = 1/np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    f15 = 0
    if len(X.shape) == 2:
        for i in range(a.shape[0]):
            f15 = f15 + (a[i]-(X[:,0]*(b[i]**2+b[i]*X[:,1])/(b[i]**2+b[i]*X[:,2]+X[:,3])))**2
    else:
        for i in range(a.shape[0]):
            f15 = f15 + (a[i]-(X[0]*(b[i]**2+b[i]*X[1])/(b[i]**2+b[i]*X[2]+X[3])))**2
    return f15
def shcb(X):
    if len(X.shape) == 2:
        f16 = 4*X[:,0]**2-2.1*X[:,0]**4+1/3*X[:,0]**6+X[:,0]*X[:,1]-4*X[:,1]**2+4*X[:,1]**4
    else:
        f16 = 4*X[0]**2-2.1*X[0]**4+1/3*X[0]**6+X[0]*X[1]-4*X[1]**2+4*X[1]**4
    return f16
def branin(X):
    if len(X.shape) == 2:
        f17 = (X[:,1]+5-X[:,0]**2*5.1/(4*np.pi**2)+5/np.pi*X[:,0]-6)**2+10*(1-1/(8*np.pi))*torch.cos(X[:,0])+10
    else:
        f17 = (X[1]+5-X[0]**2*5.1/(4*np.pi**2)+5/np.pi*X[0]-6)**2+10*(1-1/(8*np.pi))*torch.cos(X[0])+10
    return f17
def gp(X):
    if len(X.shape) == 2:
        f18 = (1+(X[:,0]+X[:,1]+1)**2*(19-14*X[:,0]+3*X[:,0]**2-14*X[:,1]+6*X[:,0]*X[:,1]+3*X[:,1]**2))*(30+(2*X[:,0]-3*X[:,1])**2*(18-32*X[:,0]+12*X[:,0]**2+48*X[:,1]-36*X[:,0]*X[:,1]+27*X[:,1]**2))
    else:
        f18 = (1+(X[0]+X[1]+1)**2*(19-14*X[0]+3*X[0]**2-14*X[1]+6*X[0]*X[1]+3*X[1]**2))*(30+(2*X[0]-3*X[1])**2*(18-32*X[0]+12*X[0]**2+48*X[1]-36*X[0]*X[1]+27*X[1]**2))
    return f18
def hartman1(X):
    a = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    c = np.array([1,1.2,3,3.2])
    p = np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    f19 = 0
    if len(X.shape) == 2:
        for i in range(a.shape[0]):
            cnt = 0
            for j in range(a.shape[1]):
                cnt = cnt - a[i][j]*(X[:,j]-p[i][j])**2
            f19 = f19 - c[i]*torch.exp(cnt)
    else:
        for i in range(a.shape[0]):
            cnt = 0
            for j in range(a.shape[1]):
                cnt = cnt - a[i][j]*(X[j]-p[i][j])**2
            f19 = f19 - c[i]*torch.exp(cnt)
    return f19
def hartman2(X):
    a = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    c = np.array([1,1.2,3,3.2])
    p = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],[0.2348,0.1451,0.3522,0.2883,0.3047,0.665],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    f20 = 0
    if len(X.shape) == 2:
        for i in range(a.shape[0]):
            cnt = 0
            for j in range(a.shape[1]):
                cnt = cnt - a[i][j]*(X[:,j]-p[i][j])**2
            f20 = f20 - c[i]*torch.exp(cnt)
    else:
        for i in range(a.shape[0]):
            cnt = 0
            for j in range(a.shape[1]):
                cnt = cnt - a[i][j]*(X[j]-p[i][j])**2
            f20 = f20 - c[i]*torch.exp(cnt)
    return f20
def shekel1(X):
    b = np.array([1,2,2,4,4,6,3,7,5,5])/10
    c = np.array([[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6],[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6]])
    f21 = 0
    if len(X.shape) == 2:
        for i in range(5):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[:,j]-c[j][i])**2
            f21 = f21 - 1/cnt
    else:
        for i in range(5):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[j]-c[j][i])**2
            f21 = f21 - 1/cnt
    return f21
def shekel2(X):
    b = np.array([1,2,2,4,4,6,3,7,5,5])/10
    c = np.array([[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6],[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6]])
    f22 = 0
    if len(X.shape) == 2:
        for i in range(7):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[:,j]-c[j][i])**2
            f22 = f22 - 1/cnt
    else:
        for i in range(7):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[j]-c[j][i])**2
            f22 = f22 - 1/cnt
    return f22
def shekel3(X):
    b = np.array([1,2,2,4,4,6,3,7,5,5])/10
    c = np.array([[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6],[4,1,8,6,3,2,5,8,6,7],[4,1,8,6,7,9,3,1,2,3.6]])
    f23 = 0
    if len(X.shape) == 2:
        for i in range(10):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[:,j]-c[j][i])**2
            f23 = f23 - 1/cnt
    else:
        for i in range(10):
            cnt = b[i]
            for j in range(4):
                cnt = cnt + (X[j]-c[j][i])**2
            f23 = f23 - 1/cnt
    return f23

# 非线性方程组优化
def F01(X):
    F01_y1 = torch.sum(X**2, dim = -1) - 1
    if len(X.shape) == 1:
        F01_y2 = torch.abs(X[0] - X[1]) + torch.sum(X[2:]**2, dim = -1)
    else:
        F01_y2 = torch.abs(X[:,0] - X[:,1]) + torch.sum(X[:, 2:]**2, dim = -1)
    return torch.sqrt(F01_y1**2 + F01_y2**2)

def F02(X):
    if len(X.shape) == 1:
        y1 = X[0] - torch.sin(5*np.pi*X[1])
        y2 = X[0] - X[1]
    else:
        y1 = X[:,0] - torch.sin(5*np.pi*X[:,1])
        y2 = X[:,0] - X[:,1]
    return torch.sqrt(y1**2 + y2**2)

def F03(X):
    if len(X.shape) == 1:
        y1 = X[0] - torch.cos(4*np.pi*X[1])
        y2 = X[0]**2 + X[1]**2
    else:
        y1 = X[:,0] - torch.cos(4*np.pi*X[:,1])
        y2 = X[:,0]**2 + X[:,1]**2
    return torch.sqrt(y1**2 + y2**2)

def F04(X):
    if len(X.shape) == 1:
        y1 = torch.cos(2*X[0]) - torch.cos(2*X[1]) - 0.4
        y2 = 2*(X[1] - X[0]) + torch.sin(2*X[1]) - torch.sin(2*X[0]) - 1.2
    else:
        y1 = torch.cos(2*X[:,0]) - torch.cos(2*X[:,1]) - 0.4
        y2 = 2*(X[1] - X[:,0]) + torch.sin(2*X[:,1]) - torch.sin(2*X[:,0]) - 1.2
    return torch.sqrt(y1**2 + y2**2)