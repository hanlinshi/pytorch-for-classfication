import torch.nn as nn
""" 定义网络 """

class Batch_Net(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的 nn.Linear
    增加了一个加快收敛速度的方法——批标准化 nn.BatchNorm1d
    在每层的输出部分添加了激活函数 nn.ReLU(True)  
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x