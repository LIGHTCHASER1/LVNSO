# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:49:52 2023

@author: 14029
"""
import torch
import networks as nw
import os
# ############################## 单峰函数 ##################################
# 简单系统会使得算法精度不够。算法的精度不够时，同等倍增vmax_eta,epsilon_start/end
# f1sphere:([200,500],200),max_epochs=6000
# f2schw222:([200,500],200),max_epochs=6000
# 收敛速度较慢的，可以减少epsilon_start/end，不过会牺牲解的精度
# 对于上下界差异大的函数，探索的vmax_eta不宜过大
# f3schw12:([20,50],2),max_epochs=6000
# f4schw221:([20,50],2),max_epochs=6000
# f5rose:([20,50],2),max_epochs=6000,效果差
# f6step:([20,50],2),max_epochs=6000
# f7quartic:([1,1],1),max_epochs=500,lr=0.05
# ############################## 多峰函数 ##################################
# f8schw:([1,1],1),max_epochs=6000
# f9ras:([20,50],20),max_epochs=6000
# f10ack:([20,50],20),max_epochs=6000
# f11grie:([1,1],1),max_epochs=6000
# f12gpf1:([20,50],2),max_epochs=500
# f13gpf2:([20,50],2),max_epochs=500
# ############################## 定维函数 ##################################
# f14foxhole,narvs=2:([20,50],2),max_epochs=500
# f15kowalik,narvs=4:([20,50],2),max_epochs=500,randombool=False
# f16shcb,narvs=2:([20,50],2),max_epochs=500,randombool=False
# f17branin,narvs=2:([20,50],2),max_epochs=500,randombool=False
# f18gp,narvs=2:([20,50],2),max_epochs=500,randombool=False
# f19hartman1,narvs=3:([20,50],2),max_epochs=500,randombool=False
# f20hartman2,narvs=6:([20,50],2),max_epochs=500,randombool=False
# f21shekel1,narvs=4:([20,50],2),max_epochs=500,randombool=False,random_particle_size = 1000000
# f22shekel2,narvs=4:([20,50],2),max_epochs=500,randombool=False,random_particle_size = 1000000
# f23shekel3,narvs=4:([20,50],2),max_epochs=500,randombool=False,random_particle_size = 1000000

# RBFs

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases

class config(object):
    def __init__(self):
        #Lstm算法参数
        self.narvs = 2
        self.hidden_size = 24
        # self.layers_num = 1
        self.dropout = 0
        self.batch_size = 50
        self.lstm_particle_size = 1000
        self.max_epochs = 10000
        self.lr = 0.0005
        self.network = nw.LinearLSTM
        #self.network = nw.LinearLSTM
        #总寻优次数
        self.particle_size = 30
        #epsilon-贪心算法参数
        self.epsilon_start = 40
        self.epsilon_end = 70
        self.epsilon_decay = self.max_epochs
        self.epsilon_bool = True
        
        #scheduler学习率衰减
        self.step_bool = True
        self.step_size = 1000
        self.step_gamma = 0.9
        #是否输出中间结果
        self.print_bool = True
        self.print_bool2 = True
        self.printx_bool = True
        #函数性质、寻优范围与目标
        self.y_aim = 0
        #LSTM粒子群初始化的范围，与原函数粒子群初始化范围无关
        self.min_x = -500
        self.max_x = 500
        
        #self.function_name = 8
        self.function_name = 'F01'
        self.bool1 = 'min'
        #是否使用PSO算法进行初始中心点寻优
        self.PSO_bool = False
        self.randomfind = True
        #随机中心点寻优粒子数
        self.random_particle_size = 1000
        #最大变化随机速度
        self.vmax_eta = 2
        #全局寻优分解系数
        self.globalsearch_phi = 0.5
        #随机概率
        self.change_opt = 0.1
        self.change_num = 1
        self.randomvbool = True
        self.circlebool = True
        
        
        self.rbfbool = False
        self.rbf_path = os.path.join('rbf_models', 'f2_0.06_20231011_023946.pth')
        self.rbf_input_dim = self.narvs
        self.rbf_output_dim = 1
        self.num_rbfs = 100
        self.rbf_basis_func = gaussian
        
        #作图参数，用于展示效果
        self.pureplot = False
        self.plotpoint = False
        self.plotiterk = 200
        self.cmap = 'winter'
        self.num_p = 15
        if self.plotpoint == True:
            self.narvs = 2
            self.lstm_particle_size = 15
def revise_cfg(cfg1,function_name):
    
    if isinstance(function_name,str):
        num = int(''.join([x for x in function_name if x.isdigit()]))
        if function_name[0] == 'f': 
            if num == 1:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta = 'f1',200,500,20
            if num == 2:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta = 'f2',200,500,200
            if num == 3:
                cfg1.function_name,cfg1.lr= 'f3',0.005
            if num == 4:
                cfg1.function_name = 'f4'
            if num == 5:
                cfg1.function_name = 'f5'
            if num == 6:
                cfg1.function_name,cfg1.max_epochs = 'f6',1000
            if num == 7:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta,cfg1.max_epochs,cfg1.lr = 'f7',1,1,1,500,0.05
            if num == 8:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta = 'f8',1,1,1
            if num == 9:
                cfg1.function_name = 'f9'
            if num == 10:
                cfg1.function_name = 'f10'
            if num == 11:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta = 'f11',1,1,1
            if num == 12:
                cfg1.function_name,cfg1.max_epochs = 'f12',500
            if num == 13:
                cfg1.function_name,cfg1.max_epochs = 'f13',500
            if num == 14:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.random_particle_size = 'f14',2,500,100000
            if num == 15:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f15',4,500,False
            if num == 16:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f16',2,500,False
            if num == 17:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f17',2,500,False
            if num == 18:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f18',2,500,False
            if num == 19:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f19',3,500,False
            if num == 20:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool = 'f20',6,500,False
            if num == 21:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool,cfg1.random_particle_size = 'f21',4,500,False,1000000
            if num == 22:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool,cfg1.random_particle_size = 'f22',4,500,False,1000000
            if num == 23:
                cfg1.function_name,cfg1.narvs,cfg1.max_epochs,cfg1.randomvbool,cfg1.random_particle_size = 'f23',4,500,False,1000000
        if function_name[0] == 'F': 
            if num == 1:
                cfg1.function_name,cfg1.epsilon_start,cfg1.epsilon_end,cfg1.vmax_eta = 'F01',200,500,20
            if num == 2:
                cfg1.function_name = 'F02'
        return cfg1
    else:
        cfg1.narvs = 1000
        return cfg1