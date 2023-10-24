# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:54:31 2023

@author: 14029
"""
# import cec2005real as cec05
import numpy as np
#from cec2013lsgo.cec2013 import Benchmark
import cfg as c
cfg = c.config()
func_num = cfg.function_name
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 原自编程函数的信息
info_dict = {
    'f1':['Sphere',-50,50,'[0,0,..,0]','0'],
    'f2':['Schw222',-10,10,'[0,0,..,0]','0'],
    'f3':['Schw12',-100,100,'[0,0,..,0]','0'],
    'f4':['Schw221',-100,100,'[0,0,..,0]'],
    'f5':['Rosenbrock',-50.0,50.0,'[1,1,..,1]','0'],
    'f6':['Step',-100.0,100.0,'[0,0,..,0]','0'],
    'f7':['Quartic',-1.28,1.28,'[0,0,..,0]','0'],
    'f8':['Schwefel',-500.0,500.0,'[420.9687,420.9687,..,420.9687]','0'],
    'f9':['Rastrigin',-5.12,5.12,'[0,0,..,0]','0'],
    'f10':['Ackley',-32.768,32.768,'[0,0,..,0]','0'],
    'f11':['Griewank',-600.0,600.0,'[0,0,..,0]','0'],
    'f12':['gpf1',-50.0,50.0,'[0,0,..,0]','0'],
    'f13':['gpf2',-50.0,50.0,'[0,0,..,0]','0'],
    'f14':['foxhole',-65.536,65.536,'[−31.97833,−31.97833]','0.998003837794449325',2],
    'f15':['kowalik',-5.0,5.0,'[0.1928,0.1908,0.1231,0.1358]','0.0003075',4],
    'f16':['shcb',-5.0,5.0,'[0.08983,-0.7126]','-1.0316285',2],
    'f17':['branin',-5.0,10.0,'[3.142,2.275]','-1.0316285',2],
    'f18':['gp',-2.0,2.0,'[0,-1]','-1.0316285',2],
    'f19':['hartman1',0.0,1.0,'[0.114,0.556,0.852]','-3.86',3],
    'f20':['hartman2',0.0,1.0,'[0.201,0.150,0.476,0.275,0.311,0.657]','-3.32',3],
    'f21':['shekel1',0.0,10.0,'[4,4,4,4]','-3.32',3],
    'f22':['shekel2',0.0,10.0,'[4,4,4,4]','-3.32',3],
    'f23':['shekel3',0.0,10.0,'[4,4,4,4]','-3.32',3],
    'F01':['F01', -1.0, 1.0, '[±0.707, ±0.707, 0,...,0]','0'],
    'F02':['F02', -1.0, 1.0, '11 roots', '0', 2]
    }
# import cec2019comp100digit as cec19
class Functions(object):
    def __init__(self,name,min_x,max_x,opx,opf):
        self.name = name
        self.min_x = min_x
        self.max_x = max_x
        self.opx = opx
        self.opf = opf
        self.narvs = 0
    def printfunctions(self):
        print("———————————————————————————————————————————————————————————————")
        print('函数名称：{}'.format(self.name))
        print('限制区间：[{},{}]'.format(self.min_x,self.max_x))
        print('{}函数对应维度全局最优解为：{}；最优值为：{}'.format(self.name,self.opx,self.opf))
        print("———————————————————————————————————————————————————————————————")
    def Engprintfunctions(self):
        print("———————————————————————————————————————————————————————————————")
        print('Function Name: {}'.format(self.name))
        print('Restrict range: [{},{}]'.format(self.min_x,self.max_x))
        print("Global best solution: {}\nGlobal best value: {}".format(self.opx,self.opf))
        print("———————————————————————————————————————————————————————————————")
#定义不定维函数的父类NonFixed
class NonFixed(Functions):
    def __init__(self,name,min_x,max_x,opx,opf):
        super().__init__(name,min_x,max_x,opx,opf)
        self.narvs = cfg.narvs
#定义定维函数的父类Fixed
class Fixed(Functions):
    def __init__(self,name,min_x,max_x,opx,opf,narvs):
        super().__init__(name,min_x,max_x,opx,opf)
        self.narvs = narvs

class get_function(object):
    def __init__(self,num,info_dic):
        if isinstance(num,str):
            self.name = num
            self.num = num
            self.info_dic = info_dic
            #self.name = info_dic[self.num][0]
            self.min_x = info_dic[self.num][1]
            self.max_x = info_dic[self.num][2]
            self.opx = info_dic[self.num][3]
            self.opf = info_dic[self.num][4]
            if len(self.info_dic[self.num]) == 5:
                self.print_func = NonFixed(self.name,self.min_x,self.max_x,self.opx,self.opf)
            else:
                if info_dic[self.num][5] != cfg.narvs:
                    print('修改cfg中维度的值，以与定维函数要求维度匹配。')
                    import sys
                    sys.exit()
                else:
                    self.narvs = cfg.narvs
                    self.print_func = Fixed(self.name,self.min_x,self.max_x,
                                               self.opx,self.opf,self.narvs)
        else:
            self.bench = Benchmark()
            self.info_dic = self.bench.get_info(num)
            self.name = 'cec2013_f{}'.format(num)
            self.min_x = self.info_dic['lower']
            self.max_x = self.info_dic['upper']
            self.opx = self.info_dic['threshold']
            self.opf = self.info_dic['best']
            if self.info_dic['dimension'] != cfg.narvs:
                print('修改cfg中维度的值，以与定维函数要求维度匹配。')
                import sys
                sys.exit()
    def printfuncinfo(self):
        self.print_func.printfunctions()
    def Engprintfuncinfo(self):
        self.print_func.Engprintfunctions()
    def calculate(self,X):
        import baseline_func as bf
        if self.name == 'f1':
            return bf.sphere(X)
        if self.name == 'f2':
            return bf.schw222(X)
        if self.name == 'f3':
            return bf.schw12(X)
        if self.name == 'f4':
            return bf.schw221(X)
        if self.name == 'f5':
            return bf.rose(X)
        if self.name == 'f6':
            return bf.step(X)
        if self.name == 'f7':
            return bf.quartic(X)
        if self.name == 'f8':
            return bf.schw(X)
        if self.name == 'f9':
            return bf.ras(X)
        if self.name == 'f10':
            return bf.ack(X)
        if self.name == 'f11':
            return bf.grix(X)
        if self.name == 'f12':
            return bf.gpf1(X)
        if self.name == 'f13':
            return bf.gpf2(X)
        if self.name == 'f14':
            return bf.foxhole(X)
        if self.name == 'f15':
            return bf.kowalik(X)
        if self.name == 'f16':
            return bf.shcb(X)
        if self.name == 'f17':
            return bf.branin(X)
        if self.name == 'f18':
            return bf.gp(X)
        if self.name == 'f19':
            return bf.hartman1(X)
        if self.name == 'f20':
            return bf.hartman2(X)
        if self.name == 'f21':
            return bf.shekel1(X)
        if self.name == 'f22':
            return bf.shekel2(X)
        if self.name == 'f23':
            return bf.shekel3(X)
        
        if self.name == 'F01':
            return bf.F01(X)
        if self.name == 'F02':
            return bf.F02(X)
        
        
        if self.name == 'cec2013_f1':
            fun_fitness = self.bench.get_function(1)
        if self.name == 'cec2013_f2':
            fun_fitness = self.bench.get_function(2)
        if self.name == 'cec2013_f3':
            fun_fitness = self.bench.get_function(3)
        if self.name == 'cec2013_f4':
            fun_fitness = self.bench.get_function(4)
        if self.name == 'cec2013_f5':
            fun_fitness = self.bench.get_function(5)
        if self.name == 'cec2013_f6':
            fun_fitness = self.bench.get_function(6)
        if self.name == 'cec2013_f7':
            fun_fitness = self.bench.get_function(7)
        if self.name == 'cec2013_f8':
            fun_fitness = self.bench.get_function(8)
        if self.name == 'cec2013_f9':
            fun_fitness = self.bench.get_function(9)
        if self.name == 'cec2013_f10':
            fun_fitness = self.bench.get_function(10)
        if self.name == 'cec2013_f11':
            fun_fitness = self.bench.get_function(11)
        if self.name == 'cec2013_f12':
            fun_fitness = self.bench.get_function(12)
        if self.name == 'cec2013_f13':
            fun_fitness = self.bench.get_function(13)
        if self.name == 'cec2013_f14':
            fun_fitness = self.bench.get_function(14)
        if self.name == 'cec2013_f15':
            fun_fitness = self.bench.get_function(15)
        #fun_fitness = np.vectorize(fun_fitness)
        X_n = X.cpu().data.numpy()
        #fv = np.vectorize(lambda x: fun_fitness(x))
        result = []
        if len(X_n.shape) == 2:
            for i in X_n:
                result.append(fun_fitness(np.float64(i)))
            return torch.from_numpy(np.array(result)).float().to(device)
        else:
            return torch.from_numpy(np.array(fun_fitness(np.float64(X_n)))).float().to(device)
        #torch.from_numpy(train_x).float().to(device)
        #选择cec2013测试的话，X目前只能是一维的，需要修改
if __name__ == '__main__':
    # f = NonFixed('Rastrigin',-5.12,5.12,'[0,0,..,0]','0',2)
    # print(f.narvs)
    # bench = Benchmark()
    # for i in range(1,16):
    #       print(bench.get_info(i))
    # fun_fitness = bench.get_function(2)
    # sol = np.float64(np.linspace(1,10,1000))
    # all_s = np.float64([sol,sol])
    # print(fun_fitness(all_s))
    gf = get_function(func_num,info_dict)
    gf.Engprintfuncinfo()