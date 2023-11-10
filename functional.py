# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:37:48 2023

@author: HUAWEI
"""

import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ith_type = 'Vanilla ES'
def BlackBoxFunction(x):
    # x is a tensor
    x = x.cpu().data.numpy()
    return torch.from_numpy(1 / (1 + np.exp(-x))).float().to(device)

def WhiteBoxFunction(x):
    # x is a tensor
    return x**4 + 1

class SurrogateGradient4ObjFunc(Function):
    
    @staticmethod
    def forward(ctx, x):
        result = BlackBoxFunction(x)
        #result = WhiteBoxFunction(x)
        ctx.save_for_backward(x,result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        x,result = ctx.saved_tensors
        #print("called")
        if ith_type == 'exp':
            update = grad_output * result
            #print(update.shape)
            #print(update)
            return update
        elif ith_type == 'Vanilla ES':
            x_dim = x.shape[0]
            sigma = 0.1
            beta = 1.0
            n = 1000
            scale = sigma / np.sqrt(n)
            epsilons = scale * torch.randn(n, 1)
            f_pos = BlackBoxFunction(x + epsilons.t())
            f_neg = BlackBoxFunction(x - epsilons.t())
            
            update = (beta / (2 * sigma ** 2)) * (f_pos - f_neg).mm(epsilons)
            #print(update.shape)
            #print(update[0])
            return update[0] * grad_output

class Network4Test(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.linear1 = nn.Linear(input_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, output_features)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = SurrogateGradient4ObjFunc.apply(x)
        return x
if __name__ == '__main__':

    # x = torch.ones(1, requires_grad=True)
    # #print(x)
    # y = SurrogateGradient4ObjFunc.apply(x)
    # y.backward()
    
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x.pow(2) + 0.2*torch.rand(x.size()) ##[]表示的是维度数据
    #神经网络只能输入Variable类型的数据
    
    #下面这两行代码可以看到神经网络生成的图长什么样子
    
    #plt.scatter(x.data.numpy(),y.data.numpy())
    #plt.show()

    net = Network4Test(1, 5, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
    loss_func = nn.MSELoss()


    for t in range(100):
        prediction =net(x)
    
        loss = loss_func(prediction,y) #//预测值一定要在前面，真实值要在后面
    
        optimizer.zero_grad() #将所有参数的梯度全部降为0，梯度值保留在这个里面
        loss.backward()    #反向传递过程
        optimizer.step()   #优化梯度
    
        if t%5==0:
            plt.cla()
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
            plt.text(0.5,0,'Loss=%.4f' % loss.data,fontdict={'size':20,'color':'red'})
            plt.pause(0.1)

    