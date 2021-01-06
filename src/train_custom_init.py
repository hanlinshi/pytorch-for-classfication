import torch.optim as optim 
from torch import nn
from torch.autograd import Variable
from loaddata import load_data
import torch
import time
import torchvision
from myvgg import vgg19_bn
from torch.nn import init


def weigth_init(net):
    '''初始化网络'''
    for m in net.modules():
        # 不同类型的层不同初始化
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)    
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1e-3)
            init.constant_(m.bias.data, 0)


def trainandsave():

    # 加载数据
    root_path = "./data/cifar10"
    data_folder = "train"
    batch_size = 32
    data_type = "train"
    trainloader = load_data(root_path, data_folder, batch_size, data_type)
    
    # 加载自定义模型 默认的分类数量就是10
    net = vgg19_bn()
    # 自定义初始化
    weigth_init(net)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.cuda()
    else:
        device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)
    net.to(device=device)
  
    # 优化器 
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)   

    # 损失函数 # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    celoss = nn.CrossEntropyLoss()

    # 训练部分
    print("trainloader = ", trainloader)
    for epoch in range(300):    # 训练的数据量为5个epoch，每个epoch为一个循环
                           
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        # 这里我们遇到了第一步中出现的trailoader，代码传入数据
        for i, data in enumerate(trainloader, 0):  
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = data  
            
            #print("inputs = ", inputs)
            #print("labels = ", labels)
            # wrap them in Variable
            # 转换数据格式用Variable
            inputs, labels = Variable(inputs), Variable(labels)   
            # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()
            # forward + backward + optimize
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 把数据输进CNN网络net
            outputs = net(inputs)        
            
            loss = celoss(outputs, labels)  # 计算损失值
            loss.backward()                    # loss反向传播 计算反向梯度
            optimizer.step()                   # 利用反向梯度 参数更新 
            #running_loss += loss.data[0]       # loss累加
            running_loss += loss.item()       # loss累加
            # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
            if (i+1) % 200 == 0:   
                localtime = time.asctime( time.localtime(time.time()) )   
                print(localtime, '[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
        # 每五十个epoch保存一次参数
        if (epoch+1) % 50 == 0:
            save_name = "custom_net_params_ep"+str(epoch) + ".pkl"
            torch.save(net.state_dict(), save_name)
            

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'pretrain_custom_net.pkl') # 保存整个神经网络的结构和模型参数
    # torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数


if __name__ == "__main__":
    trainandsave()
