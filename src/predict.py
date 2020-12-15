# -*-coding:utf-8 -*-
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def detect_one_image():

    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = "net.pkl"
    net=torch.load(model_path)

    # === 读取图像
    image_file = "data/data_batch_1_30.png"
    image = Image.open(image_file).convert('RGB')

    # === 数据处理
    # 转换成tensor
    image = transforms.ToTensor()(image) # 转换成tensor
    # 标准化， 与训练时的处理保持一致
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])(image)
    # 增加一个维度
    image = image.unsqueeze(0)
    # 图像数据计算设备设置
    image = image.to(device)

    # 预测识别
    outputs = net(image)
    _, preds = torch.max(outputs, 1)
 
    # 结果翻译成标签名字
    label_name_path = "./data/cifar-10-batches-py/batches.meta"
    label_names = unpickle(label_name_path)
    label_names = label_names[b'label_names']
    label = label_names[preds[0].item()]
    label = label.decode("utf-8")  # bytes 转 str
    # 打印结果
    print(label)

if __name__ == "__main__":
    detect_one_image()