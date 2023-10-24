import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import test_functions as tfu
import datetime
def create_FatherFolder(save_path):   
    # 创建父目录中的文件夹（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
def train_save_model(save_path):
    create_FatherFolder(save_path)
    function_name = 'f1'
    samples = 20000
    xdim = 2
    rbf = RBFModels_trainer()
    X,y = dataset_setter(function_name, samples, xdim)
    rbf.train(X, y)
    rbf.save(save_path, function_name)

def load_test_model(load_path):
    rbfm = RBFModels_trainer()
    rbfm.rbf_net.load_state_dict(torch.load(load_path))
    function_name = 'f1'
    xdim = 2
    gf = tfu.get_function(function_name, tfu.info_dict)
    minx = gf.min_x
    maxx = gf.max_x
    test_tx = torch.from_numpy(np.random.uniform(minx, maxx, (5, xdim))).float()
    test_ty = rbfm.rbf_net(test_tx)
    ty = gf.calculate(test_tx).view(-1, 1)
    print(test_ty.detach().numpy())
    print(ty.detach().numpy())
    return rbfm.criterion(ty, test_ty)

class RBFNetwork(nn.Module):
    def __init__(self, input_dim, num_rbfs, output_dim, basis_func):
        super(RBFNetwork, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_rbfs, input_dim))
        self.sigmas = nn.Parameter(torch.ones(num_rbfs, 1))
        self.linear = nn.Linear(num_rbfs, output_dim)
        self.basis_func = basis_func

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        norm = torch.norm(diff, dim=2)
        phi = norm.pow(2) / (self.sigmas.T ** 2)
        output = self.linear(self.basis_func(phi))
        return output


class RBFModels_trainer():
    def __init__(self):
        num_rbfs = 100
        output_dim = 1
        input_dim = 2
        self.step = 20000
        self.batch = 64
        self.lr = 0.5
        self.num_rbfs = 100
        self.basis_func = gaussian
        self.rbf_net = RBFNetwork(input_dim, num_rbfs, output_dim, self.basis_func)
        self.criterion = nn.MSELoss()

    def train(self, X, y):
        y = y.to(torch.float32)

        
        optimizer = optim.Adam(self.rbf_net.parameters(), lr=self.lr)
        print("train_device = cuda" if torch.cuda.is_available() else "train_device = cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        self.rbf_net.to(device)
        y = y.to(device)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.9, last_epoch=-1)

        for epoch in range(self.step):
            optimizer.zero_grad()
            predictions = self.rbf_net(X).to(torch.float32)
            loss = self.criterion(predictions, y)
            loss.backward()
            optimizer.step()
            schedule.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{self.step}], Loss: {loss.item():.4f}')  # 定义损失函数和优化器
        self.losses = loss.item()

    def save(self, save_path, function_name):
        create_FatherFolder(save_path)
        # 获取当前时间的字符串表示（精确到秒）
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'{function_name}_{self.losses:.2f}_{current_time}.pth'
        save_path = os.path.join(save_path, model_name)
        torch.save(self.rbf_net.state_dict(), save_path)
        
    def predict(self, X_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test = X_test.to(device)
        return self.rbf_net(X_test)

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


def levy13(X):
    d = X.shape[-1]
    f7 = 0
    if len(X.shape) == 2:
        for i in range(d):
            f7 = f7 + i * X[:, i] ** 4
    else:
        for i in range(d):
            f7 = f7 + i * X[i] ** 4
    return f7   # + np.random.rand(X.shape[0])

def dataset_setter(function_name, samples, xdim):
    if function_name != 'others':
        gf = tfu.get_function(function_name, tfu.info_dict)
        minx = gf.min_x
        maxx = gf.max_x
    xs = np.random.uniform(minx, maxx, (samples, xdim))
    tx = torch.from_numpy(xs).float()
    ty = gf.calculate(tx).view(-1, 1)
    return tx, ty

if __name__ == "__main__":
    # # 生成 x1 和 x2 的范围
    # x1 = torch.linspace(-5, 5, 50)
    # x2 = torch.linspace(-5, 5, 50)
    # x3 = torch.linspace(-5, 5, 50)
    
    
    # # 创建一个网格以计算对应的 y 值
    # X1, X2, X3 = np.meshgrid(x1, x2, x3)
    # X1 = X1.reshape(1, -1).T
    # X2 = X2.reshape(1, -1).T
    # X3 = X3.reshape(1, -1).T
    # X = np.column_stack((X1, X2, X3))
    # Y = np.zeros((X.shape[0], 1))
    
    # Y = torch.from_numpy(levy13(X).reshape(-1, 1))
    
    save_path = 'rbf_models'
    #train_save_model(save_path)
    
    load_path = os.path.join(save_path, 'f1_7.24_20231011_012548.pth')
    load_test_model(load_path)





