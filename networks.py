# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:51:42 2023

@author: 14029
"""

import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import LSTM, Linear, MSELoss, Parameter, Dropout, Module
import random
import math
import numpy as np
from torch import sin,exp,cos,log,sqrt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
from functools import reduce
torch.autograd.set_detect_anomaly = True

class ReluLSTM(Module):
    # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # change tanh into relu
    def __init__(self, input_size, hidden_size):
        super(ReluLSTM, self).__init__()
        self.h0 = Parameter(torch.randn(hidden_size))
        self.c0 = Parameter(torch.randn(hidden_size))

        self.linear1 = Linear(input_size + hidden_size, hidden_size)
        self.linear2 = Linear(input_size + hidden_size, hidden_size)
        self.linear3 = Linear(input_size + hidden_size, hidden_size)
        self.linear4 = Linear(input_size + hidden_size, hidden_size)

    def forward(self, input_data, batch_first=False):
        # input_data = (batch, seq_len, input_size)
        # 一个 (batch,序列长度,每个data的维度) 大小的输入
        # input_data (batch, input_size) input in a time step
        if batch_first:
            input_data = input_data.permute(1, 0, 2) # 将第二维与第一维换位
        batch_size = input_data.size(1) # 记录第二维的数据作为batch_size
        seq_len = input_data.size(0) # 记录第一维的数据作为seq_len序列长度
        ht = self.h0.unsqueeze(0).repeat(batch_size, 1)
        ct = self.c0.unsqueeze(0).repeat(batch_size, 1)
       
        # .unsqueeze(0)可将h0在已有维度上再加一个维度，即[i,j,k] → [[i,j,k]]
        # .repeat(batch_size, 1)可将其在新的维度上重复batch_size遍
        # (不用unsqueeze(0)也能达到这一效果)
        """
        torch.tensor([1, 2, 3]).unsqueeze(0).repeat(4,1)
        与
        torch.tensor([1, 2, 3]).repeat(4,1)效果相同，
        均为torch.Size([4, 3])的向量：
        tensor([[1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]])
        """
        
        for i in range(seq_len):
            # 这里使用seq_len循环遍历，每次遍历训练一个batch的所有元素
            # 对每个batch而言，里面有batch_size个训练的input_data
            ft = torch.sigmoid(self.linear1(torch.concat([ht, input_data[i]], dim=-1)))
            it = torch.sigmoid(self.linear2(torch.concat([ht, input_data[i]], dim=-1)))
            # 按照最后一维dim = -1，也就是input_size维度将ht与input_data进行合并
            # 作为下一步的输入
            ct_ = torch.tanh(self.linear3(torch.concat([ht, input_data[i]], dim=-1)))
            ct = ft * ct + it * ct_ # LSTM参数隐藏层更新（细胞层）
            ot = torch.sigmoid(self.linear4(torch.concat([ht, input_data[i]], dim=-1)))
            ht = ot * torch.tanh(ct) # LSTM的t时刻输出ht

        # ht (batch, hidden_size), ct(batch, hidden_size)
        return ht, ct

# 各种网络效果尝试
class LinearLSTM(Module):
    def __init__(self, input_size,hidden_size,output_size,dropout_param):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size,hidden_size,1)#单层LSTM效果较好，固定为1层
        self.output_linear = nn.Linear(hidden_size,output_size)
        #self.drop_out = Dropout(p=dropout_param)
        
    def forward(self, x):
        x, _ = self.lstm(x)#_x is input, size (seq_len, batch, input_size)
        #x = self.drop_out(x)
        x = self.output_linear(x)
        return x

class LinearReluLSTM(Module):
    def __init__(self, input_size,hidden_size,output_size,dropout_param):
        super().__init__()
        
        self.input_linear = Linear(input_size,hidden_size)
        self.lstm = ReluLSTM(hidden_size,hidden_size)
        self.output_linear = Linear(hidden_size,output_size)
        self.drop_out = Dropout(p=dropout_param)
        
    def forward(self, x):
        x = self.input_linear(x)
        x, _ = self.lstm(x)#_x is input, size (seq_len, batch, input_size)
        x = self.drop_out(x)
        x = self.output_linear(x)
        return x

class DoubleLinearLSTM(Module):
    def __init__(self, input_size,hidden_size,output_size,dropout_param):
        super().__init__()
        
        self.input_linear = Linear(input_size,hidden_size)
        self.lstm = LSTM(hidden_size,hidden_size,1)
        self.output_linear = Linear(hidden_size,output_size)
        self.drop_out = Dropout(p=dropout_param)
        
    def forward(self, x):
        x = self.input_linear(x)
        x, _ = self.lstm(x)#_x is input, size (seq_len, batch, input_size)
        x = self.drop_out(x)
        x = self.output_linear(x)
        return x

class DoubleLinearReluLSTM(Module):
    def __init__(self, input_size,hidden_size,output_size,dropout_param):
        super().__init__()
        
        self.input_linear = Linear(input_size,hidden_size)
        self.lstm = ReluLSTM(hidden_size,hidden_size)
        self.output_linear = Linear(hidden_size,output_size)
        self.drop_out = Dropout(p=dropout_param)
        
    def forward(self, x):
        x = self.input_linear(x)
        x, _ = self.lstm(x)#_x is input, size (seq_len, batch, input_size)
        x = self.drop_out(x)
        x = self.output_linear(x)
        return x

def BlackBoxFunction(y_now, y_aim, function_name):
    import test_functions as tfu
    gf = tfu.get_function(function_name,tfu.info_dict)
    y = gf.calculate(x_now)
    return torch.sqrt(nn.MSELoss(y_now - y_aim))

class get_BBF(object):
    def __init__(self, y_aim, function_name):
        self.y_aim = y_aim
        self.function_name = function_name
        import test_functions as tfu
        self.gf = tfu.get_function(self.function_name,tfu.info_dict)
        self.loss = nn.MSELoss()
    def BlackBoxFunction(self, x_now):
        y = self.gf.calculate(x_now)
        return torch.sqrt(self.loss(y , self.y_aim))
    
import cfg as c 
cfgs = c.config()
gBBF = get_BBF(cfgs.y_aims, 'f8')
ith_type = 'Vanilla ES'

class SurrogateGradient4ObjFunc(Function):
    
    @staticmethod
    def forward(ctx, x):
        #result = gBBF.BlackBoxFunction(x)
        #result = WhiteBoxFunction(x)
        ctx.save_for_backward(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        x = ctx.saved_tensors
        #print("called")
        if ith_type == 'Vanilla ES':
            x_dim = x.shape[0]
            sigma = 0.1
            beta = 1.0
            n = 1000
            scale = sigma / np.sqrt(n)
            epsilons = scale * torch.randn(n, 1)
            f_pos = gBBF.BlackBoxFunction(x + epsilons.t())
            f_neg = gBBF.BlackBoxFunction(x - epsilons.t())
            
            update = (beta / (2 * sigma ** 2)) * (f_pos - f_neg).mm(epsilons)
            #print(update.shape)
            #print(update[0])
            return update[0] * grad_output
        
class SurrogateGradientLinearLSTM(Module):
    def __init__(self, input_size,hidden_size,output_size,dropout_param):
        super().__init__()
        import cfg as c
        import functional as f
        cf1 = c.config()
        
        self.lstm = nn.LSTM(input_size,hidden_size,1)#单层LSTM效果较好，固定为1层
        self.output_linear = nn.Linear(hidden_size,output_size)
        #self.drop_out = Dropout(p=dropout_param)
        
    def forward(self, x):
        x, _ = self.lstm(x)#_x is input, size (seq_len, batch, input_size)
        #x = self.drop_out(x)
        x = self.output_linear(x)
        x = SurrogateGradient4ObjFunc.apply(x)
        return x