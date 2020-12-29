import torch.optim as optim 
from torch import nn
from torch.autograd import Variable
from loaddata import load_data
from mylenet import LeNet
from myvgg import vgg16_bn
from mysimplenet import Batch_Net
import torch
from tensorboardX import SummaryWriter
import time

def trainandsave():

    # 加载数据
    root_path = "./data/cifar10"
    data_folder = "train"
    batch_size = 64
    data_type = "train"
    trainloader = load_data(root_path, data_folder, batch_size, data_type)
    
    # 神经网络结构
    # 输入是32*32*3=3072维度, 中间层分别是1500, 200, 输出10个维度(10个分类)
    
    # net = Batch_Net(32*32, 1500, 200, 10)

    
    # net = vgg16_bn()


    net = LeNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)
    net.to(device=device)
    
    # 优化器 # 学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   
    # 损失函数 # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    celoss = nn.CrossEntropyLoss()

    # 训练部分
    print("trainloader = ", trainloader)
    writer = SummaryWriter('runs/train2')  # 记录
    for epoch in range(250):    # 训练的数据量为5个epoch，每个epoch为一个循环
                           
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
            
            # inputs 需要从32*32的图像展开成1024

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
                writer.add_scalar('running_loss', running_loss / 200, global_step=((epoch*600) + (i + 1)))                       
                print(localtime, '[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
        # 每50个epoch保存一次参数
        if (epoch+1) % 50 == 0:
            save_name = "net_params"+str(epoch+1) + ".pkl"
            torch.save(net.state_dict(), save_name)

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')                      # 保存整个神经网络的结构和模型参数
    # torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数


if __name__ == "__main__":
    trainandsave()