# -*- coding:utf-8 -*-
"""1 加载数据""" 
import torchvision
import torch
import torchvision.transforms as transforms
import os

def load_data(root_path, data_type, batch_size, phase):
    # transforms.Compose()类用来组合多个torchvision.transforms操作
    # 一个list数组，数组里是多个'Transform'对象
    # 遍历list数组，对img依次执行每个transforms操作，并返回transforms后的img

    # RandomResizedCrop 将给定图像随机裁剪为不同的大小和宽高比,然后缩放所裁剪得到的图像为制定的大小
    # RandomHorizontalFlip 以给定的概率随机水平翻转给定的PIL图像
    # ToTensor 把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型
    # Normalize  对每个通道而言执行操作image=(image-mean)/std

    transform_dict = {
        'train': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'test': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}

    # torchvision 的 ImageFolder 能以目录名为标签,对数据集做划分
    # 其成员变量 self.classes 保存类名的list. self.class_to_idx类名对应的索引.
    # self.imgs 保存(img_path,class)tuple的list
    # /root/dog/image1
    # /root/dog/image2
    # ....
    # /root/cat/image1
    # /root/cat/image2
    # ....
    data = torchvision.datasets.ImageFolder(root=os.path.join(root_path,data_type), transform=transform_dict[phase])
    # DataLoader 构建可迭代的数据装载器
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader 