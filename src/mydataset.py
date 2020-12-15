# -*- coding:utf-8 -*-
from PIL import Image
from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, root_path, transform=None):
        imgs = []
        for label in os.listdir(root_path):
            for imname in os.listdir(os.path.join(root_path, label)):
                imfile = os.path.join(root_path, label, imname)
                imgs.append(imfile, label)

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)