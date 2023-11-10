# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:03:25 2023

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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import math
import numpy as np
from torch import sin,exp,cos,log,sqrt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
from functools import reduce
from RBFN import RBFN
import rbfn_all as rbf

import hurbf as hrbf

torch.autograd.set_detect_anomaly = True

import cfg as c
class PSOL(object):
    def __init__(self,cfg):
        self.narvs = cfg.narvs
        self.input_features_num = cfg.narvs
        self.hidden_size = cfg.hidden_size
        self.output_features_num = cfg.narvs
        self.dropout = cfg.dropout
        self.max_epochs = cfg.max_epochs
        self.batch_size = cfg.batch_size
        self.lstm_particle_size = cfg.lstm_particle_size
        self.lr = cfg.lr
        self.step_bool = cfg.step_bool
        self.step_size = cfg.step_size
        self.step_gamma = cfg.step_gamma
        
        self.particle_size = cfg.particle_size
        self.print_bool = cfg.print_bool
        self.print_bool2 = cfg.print_bool2
        self.printx_bool = cfg.printx_bool
        self.bool1 = cfg.bool1
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon_bool = cfg.epsilon_bool
        
        self.multi_root_num = cfg.multi_root_num
        self.multi_xbest_jumpbool = cfg.multi_xbest_jumpbool
        self.group_num = int(cfg.lstm_particle_size / cfg.multi_root_num)
        
        self.rbfbool = cfg.rbfbool
        self.input_dim = cfg.rbf_input_dim
        self.output_dim = cfg.rbf_output_dim
        self.num_rbfs = cfg.num_rbfs
        self.basis_func = cfg.rbf_basis_func
        
        self.y_aim = cfg.y_aim
        self.min_x = cfg.min_x
        self.max_x = cfg.max_x
        
        self.function_name = cfg.function_name
        self.PSO_bool = cfg.PSO_bool
        self.random_particle_size = cfg.random_particle_size
        #lstm神经网络初始化
        #Initializing lstm parameters
        lstm_v = self.min_x+(self.max_x-self.min_x)*np.random.rand(self.lstm_particle_size,self.narvs)
        train_x = lstm_v.reshape(-1,self.batch_size,self.input_features_num)
        self.train_x_tensor = torch.from_numpy(train_x).float().to(device)
        self.y_tensor1 = self.y_aim*torch.ones(self.lstm_particle_size).to(device)
        
        self.lstm_model = cfg.network(self.input_features_num,self.hidden_size,self.output_features_num,self.dropout).to(device)
        self.optimizer = torch.optim.Adam(self.lstm_model.parameters(),lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size = self.step_size, gamma = self.step_gamma)
        self.Loss = nn.MSELoss()
        if self.rbfbool == True:
            self.rbf_path = cfg.rbf_path
            self.rbfn = hrbf.RBFNetwork(self.input_dim, self.num_rbfs, 
                                        self.output_dim, self.basis_func)
            self.rbfn.load_state_dict(torch.load(self.rbf_path))
        
        
        self.vmax_eta = cfg.vmax_eta
        self.globalsearch_phi = cfg.globalsearch_phi
        self.randomvbool = cfg.randomvbool
        self.change_opt = cfg.change_opt
        self.change_num = cfg.change_num
        
        self.randomfind = cfg.randomfind
        self.circlebool = cfg.circlebool
        
        self.pureplot = cfg.pureplot
        self.plotpoint = cfg.plotpoint
        self.plotiterk = cfg.plotiterk
        self.cmap = cfg.cmap
        self.num_p = cfg.num_p
    
    def PSOL_Optimizer(self,center):
        import test_functions as tfu
        gf = tfu.get_function(self.function_name,tfu.info_dict)
        minx = gf.min_x
        maxx = gf.max_x
        self.vmax = self.vmax_eta * (self.max_x-self.min_x)
        if self.randomfind == True:
            xs = minx+(maxx-minx)*np.random.rand(self.random_particle_size,self.narvs)
            #print(x_best)
            xs = torch.from_numpy(xs).float()
            y = gf.calculate(xs)
            #print(y)
            a = torch.argmin(y)
            x_best = xs[a,:].to(device)
            self.random_x_best = x_best
            self.random_y = torch.min(y)
            print('random search completed')
            #print(x_best)
        else:
            x_best = center
            self.random_y = gf.calculate(x_best)
        lost = []
        # x_1 为上一时刻的位移
        # x_1 = self.lstm_model(self.train_x_tensor.clone()).view(-1,self.output_features_num) 
        for epoch in range(0,self.max_epochs):
            # if epoch == 0:
            #     output =self.lstm_model(self.train_x_tensor.clone()).view(-1,self.output_features_num) 
            # else:
            #     output =self.lstm_model((fx_tensor_out).detach().reshape(-1,self.batch_size,self.input_features_num)).view(-1,self.output_features_num)
            
            output =self.lstm_model(self.train_x_tensor.clone()).view(-1,self.output_features_num) 
            
            if self.epsilon_bool == True:
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.ma.exp(-1. * epoch / self.epsilon_decay)
                #output = output/self.epsilon
                if epoch < self.globalsearch_phi * self.max_epochs and self.randomvbool:
                    for i in range(output.shape[0]):
                        r = np.random.uniform(0,1)
                        if r <= self.change_opt:
                            for k in range(0,self.change_num):
                                j = int(np.random.choice(np.linspace(0,self.output_features_num-1,self.output_features_num)))
                                #output[i] = torch.from_numpy(np.random.uniform(-self.vmax/self.epsilon,self.vmax/self.epsilon,self.output_features_num)).float().to(device)
                                output[i][j] = (np.random.uniform(-self.vmax/self.epsilon,self.vmax/self.epsilon))
            if epoch > 0.1*self.max_epochs and self.circlebool == True and center == None:
                import time
                time.sleep(1)
                print('Begin resetting the net......')
                conf = c.config()
                cfg1 = c.revise_cfg(conf,self.function_name)
                cfg1.randomfind,cfg1.PSO_bool,cfg1.circlebool,cfg1.randomvbool = False,False,False,False
                psol = PSOL(cfg1)
                return psol.PSOL_Optimizer(x_best)
            m1x = self.min_x-x_best.detach()
            m2x = self.max_x-x_best.detach()
            # print(output.shape)
            # print(m1x.shape)
            output = torch.clamp(output,min=m1x,max=m2x)
            # 加leader
            if self.rbfbool == False:
                fx_tensor_out = gf.calculate(output+x_best.detach())
            else:
                fx_tensor_out = self.rbfn(output+x_best.detach())
            
            # 加最初位移
            # fx_tensor_out = gf.calculate(output+self.train_x_tensor.clone().view(-1,self.output_features_num))
            # 加上一时刻位移
            # fx_tensor_out = gf.calculate(output+x_1)
            # x_1 = output + x_1
            a = torch.argmin(fx_tensor_out)
            if gf.calculate(output[a,:]+x_best) < gf.calculate(x_best):
                x_best = output[a,:] + x_best
            self.optimizer.zero_grad()
            
            loss = torch.sqrt(self.Loss(fx_tensor_out,self.y_tensor1))
            # 原算法  
            #loss = torch.sum(torch.sqrt(Loss(output,output[a,:])))
            #print(loss)
            #loss = Loss(fx_tensor_out,self.y_tensor1)
            #print(loss)
            #loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            
            self.optimizer.step()
            if self.step_bool:
                self.scheduler.step()
            
            lost.append(loss.cpu().data.numpy())
            
            if (epoch+1) % 50 == 0:
                if self.print_bool:
                    print('Epoch: [{}/{}], Loss:{:.8f}'.format(epoch+1, self.max_epochs, loss.item()))
                    if self.printx_bool:
                        print((x_best).cpu().data.numpy())
                    print(gf.calculate(x_best).cpu().data.numpy())
            
            if epoch == self.max_epochs - 1:
                f_best = gf.calculate(x_best).cpu().data.numpy()
                #print((output+x_best).cpu().data.numpy())
                x_best = (x_best).cpu().data.numpy()
                if self.print_bool2:
                    # print('随机算法最优解为：{}'.format(self.random_x_best))
                    # print('随机算法最优值为：{}'.format(self.random_y))
                    # if self.bool1 == 'min':
                    #     eta = (self.random_y.cpu().data.numpy()-f_best)/np.abs(self.random_y.cpu().data.numpy())
                    #     print('优化比例为：{}'.format(eta))
                    print('最优解为：{}'.format(x_best))
                    print('最优值为：{}'.format(f_best))
            
        x = np.arange(self.max_epochs)
        plt.plot(x,lost)
        plt.show()
        return np.append(x_best,f_best)
    
    def MultiRoot_PSOL_Optimizer(self, center):
        import test_functions as tfu
        gf = tfu.get_function(self.function_name, tfu.info_dict)
        self.vmax = self.vmax_eta * (self.max_x - self.min_x)
        minx = gf.min_x
        maxx = gf.max_x
        if self.randomfind == True:
            minx = gf.min_x
            maxx = gf.max_x
            '''
            xs: (particle_size, narvs)
            '''
            xs = minx + (maxx - minx)*np.random.rand(self.random_particle_size, self.narvs)
            xs = torch.from_numpy(xs).float()
            y = gf.calculate(xs)
            sorted_indices = torch.topk(y, self.multi_root_num, largest = False).indices
            multi_xbest = xs[sorted_indices,:].to(device)
            self.random_x_best = multi_xbest
            self.random_y = y[sorted_indices]
            print('random search completed')
        else:
            multi_xbest = center
            self.random_y = gf.calculate(multi_xbest)
        lost = []
        for epoch in range(0,self.max_epochs):
            output =self.lstm_model(self.train_x_tensor.clone()).view(-1,self.output_features_num) 
            if self.epsilon_bool == True:
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.ma.exp(-1. * epoch / self.epsilon_decay)
                #output = output/self.epsilon
                if epoch < self.globalsearch_phi * self.max_epochs and self.randomvbool:
                    for i in range(output.shape[0]):
                        r = np.random.uniform(0,1)
                        if r <= self.change_opt:
                            for k in range(0,self.change_num):
                                j = int(np.random.choice(np.linspace(0,self.output_features_num-1,self.output_features_num)))
                                #output[i] = torch.from_numpy(np.random.uniform(-self.vmax/self.epsilon,self.vmax/self.epsilon,self.output_features_num)).float().to(device)
                                output[i][j] = (np.random.uniform(-self.vmax/self.epsilon,self.vmax/self.epsilon))
            if epoch > 0.1*self.max_epochs and self.circlebool == True and center == None:
                import time
                time.sleep(1)
                print('Begin resetting the net......')
                conf = c.config()
                cfg1 = c.revise_cfg(conf,self.function_name)
                cfg1.randomfind,cfg1.PSO_bool,cfg1.circlebool,cfg1.randomvbool = False,False,False,False
                psol = PSOL(cfg1)
                return psol.MultiRoot_PSOL_Optimizer(multi_xbest)  
            #若共有1000个particle，假设20个根，那么每个组就有50个particle，分别寻优
            #接下来作每个group的速度矩阵clamp限制
            for i in range(self.multi_root_num):
                m1x = minx - multi_xbest[i,:].detach()
                m2x = maxx - multi_xbest[i,:].detach()
                # 记得在操作前.clone()，避免原位操作！！！！！！！！！！！
                output[int(i*self.group_num):int((i+1)*self.group_num), :] = torch.clamp(output[int(i*self.group_num):int((i+1)*self.group_num), :].clone(),min=m1x,max=m2x)

            # multi leaders
            # print(output.shape)
            # loss = torch.sqrt(self.Loss(output,self.y_tensor1.unsqueeze(1).expand(-1,2)))
            # with torch.autograd.detect_anomaly(True):
            #     loss.backward(retain_graph=True)

            if self.rbfbool == False:
                multi_fx_tensor_out = gf.calculate(output+torch.tile(multi_xbest.detach(), (1, self.group_num)).reshape(-1,self.narvs).detach())
            else:
                multi_fx_tensor_out = self.rbfn(output+torch.tile(multi_xbest, (1, self.group_num)).reshape(-1,self.narvs).detach())
            for i in range(self.multi_root_num):
                #print(multi_fx_tensor_out.shape)
                a = torch.argmin(multi_fx_tensor_out[int(i*self.group_num):int((i+1)*self.group_num)])
                if gf.calculate(output[a,:]+multi_xbest[i,:]) < gf.calculate(multi_xbest[i,:]):
                    if self.multi_xbest_jumpbool == False:
                        multi_xbest[i,:] = output[a,:].detach() + multi_xbest[i,:].detach()
                    else:
                        mindis = tfu.min_generalized_dis(i,minx, maxx, multi_xbest)
                        if mindis != False:
                            newdis = torch.sqrt(torch.sum(output[a,:]**2))
                            if newdis <= mindis:
                                multi_xbest[i,:] = output[a,:].detach() + multi_xbest[i,:].detach()
                            
            self.optimizer.zero_grad()
            loss = torch.sqrt(self.Loss(multi_fx_tensor_out,self.y_tensor1))
            with torch.autograd.detect_anomaly(True):
                loss.backward(retain_graph=True)
            self.optimizer.step()
            if self.step_bool:
                self.scheduler.step()
            lost.append(loss.cpu().data.numpy())
            
            if (epoch+1) % 50 == 0:
                if self.print_bool:
                    print('Epoch: [{}/{}], Loss:{:.8f}'.format(epoch+1, self.max_epochs, loss.item()))
                    if self.printx_bool:
                        print((multi_xbest).cpu().data.numpy())
                    print(gf.calculate(multi_xbest).cpu().data.numpy())
            if epoch == self.max_epochs - 1:
                f_best = gf.calculate(multi_xbest).cpu().data.numpy()
                #print((output+x_best).cpu().data.numpy())
                multi_xbest = (multi_xbest).cpu().data.numpy()
                if self.print_bool2:
                    # print('随机算法最优解为：{}'.format(self.random_x_best))
                    # print('随机算法最优值为：{}'.format(self.random_y))
                    # if self.bool1 == 'min':
                    #     eta = (self.random_y.cpu().data.numpy()-f_best)/np.abs(self.random_y.cpu().data.numpy())
                    #     print('优化比例为：{}'.format(eta))
                    print('最优解为：{}'.format(multi_xbest))
                    print('最优值为：{}'.format(f_best))
        x = np.arange(self.max_epochs)
        plt.plot(x,lost)
        plt.show()
        return np.append(multi_xbest,f_best)
    
    
    
    def PSOL_Optimizer_Epochs(self,cfg):
        times_rational = np.array([])
        for i in range(self.particle_size):
            psol = PSOL(cfg)
            psolo = psol.PSOL_Optimizer(None)
            if times_rational.shape[0]==0:
                cnt = 0
                for j in range(self.narvs):
                    if psolo[:-1][j]<psol.maxx and psolo[:-1][j]>psol.minx:
                        cnt = cnt + 1
                if cnt == self.narvs:
                    times_rational = np.append(times_rational,psolo)
            else:
                cnt = 0
                for j in range(self.narvs):
                    if psolo[:-1][j]<psol.maxx and psolo[:-1][j]>psol.minx:
                        cnt = cnt + 1
                if cnt == self.narvs:
                    times_rational = np.row_stack((times_rational,psolo))
            print('Epoch: [{}/{}],function:{}'.format(i+1, self.particle_size,self.function_name))

        if self.bool1 == 'min':
            self.f_globalbest = min(times_rational[:,-1])
            self.f_argminbest = np.argmin(times_rational[:,-1])
        if self.bool1 == 'max':
            self.f_globalbest = max(times_rational[:,-1])
            self.f_argminbest = np.argmin(times_rational[:,-1])
        self.x_globalbest = times_rational[:,:-1][self.f_argminbest]
        print('')
        print('全局最优解为：{}'.format(self.x_globalbest))
        print('全局最优值为：{}'.format(self.f_globalbest))
        print('迭代解在范围内的概率为：{}'.format(times_rational.shape[0]/self.particle_size))
        plt.show()
        return times_rational[:,-1]

def test(func_name,epochsbool):
    conf = c.config()
    cfg1 = c.revise_cfg(conf,func_name)
    print(cfg1.function_name)
    psol = PSOL(cfg1)
    if epochsbool == True:
        test_array = psol.PSOL_Optimizer_Epochs(cfg1)
        print(PSOL.func_name)
        return test_array
    else:
        print(psol.function_name)
        # psol.PSOL_Optimizer(None)
        psol.MultiRoot_PSOL_Optimizer(None)


if __name__ == '__main__':
    test('F02',False)
    #test('f9',False)