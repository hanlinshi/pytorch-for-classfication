import torch.optim as optim 
from torch import nn
from torch.autograd import Variable
from loaddata import load_data
import pickle

import torch
def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def testmodel():

    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    root_path = "./data/cifar10_save"
    data_folder = "test"
    batch_size = 8
    data_type = "test"
    testloader = load_data(root_path, data_folder, batch_size, data_type)

    # 加载模型
    model_path = "net.pkl"
    net=torch.load(model_path)
    # 选择模式 测试模式 
    # 这是因为这两个方法是针对在网络训练和测试时
    # 采用不同方式的情况，比如Batch Normalization 和 Dropout
    net.eval()

    # 加载标签名字
    label_name_path = "./data/cifar-10-batches-py/batches.meta"
    labelnames = unpickle(label_name_path)
    labelnames = labelnames[b'label_names']
    label_names = []
    for l in labelnames:
        label_names.append(l.decode("utf-8"))
    # 识别出来的数量
    pred_num = dict()
    # 标签总数量
    test_num = dict()
    # 识别对的数量
    correct_num = dict()
    for label in label_names:
        test_num[label] = 0
        pred_num[label] = 0
        correct_num[label] = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(testloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))

            image = image.to(device)
            label = label.to(device)

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            labels = label.cpu().numpy()
                # test_num[l] += 1
            for p in pred.cpu().numpy():
                pred_num[label_names[p[0]]] += 1
            for i, c in enumerate(correct.cpu().numpy()):
                l = labels[i][0]
                label_name = label_names[l]
                test_num[label_name] += 1
                if int(c[0]) == 1:
                    correct_num[label_name] += 1

    print("=====")
    print(" label name  recall  precision")
    for i in range(10):
        label_name = label_names[i]
        print(label_name, " {:.1f}%  {:.1f}%".format(float(correct_num[label_name])/float(test_num[label_name])*100, float(correct_num[label_name])/float(pred_num[label_name])*100))
       
    

if __name__ == "__main__":
    testmodel()