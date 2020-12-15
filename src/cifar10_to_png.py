# -*- coding:utf-8 -*-
"""
将cifar10数据集提取出来
放入到标签命名的文件夹中
"""

import os
import pickle
import numpy as np
import cv2


def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

def main():

    # 10标签
    label_name_path = "./data/cifar-10-batches-py/batches.meta"
    label_names = unpickle(label_name_path)
    label_names = label_names[b'label_names']
    
    # 构建图片保存目录
    root_path = "data/cifar10"
    my_mkdir(root_path)
    for foldername in label_names:
        foldername = foldername.decode("utf-8")  # bytes 转 str
        label_path = os.path.join(root_path, foldername)
        my_mkdir(label_path)
    

    # 解析数据文件
    # 数据文件名称
    data_folder_names = ["data_batch_1", "data_batch_2", "data_batch_3",
                         "data_batch_4", "data_batch_5", "test_batch"]
    for foldername in data_folder_names:
        print(foldername, " is loading...")
        data_path = os.path.join("./data/cifar-10-batches-py/", foldername)
        data_dict = unpickle(data_path)
        # 每个数据批都有1万个样本
        for i in range(0, 10000):
            img = np.reshape(data_dict[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)

            label_num = data_dict[b'labels'][i]
            label = label_names[label_num].decode("utf-8")

            im_name = foldername + "_" + str(i) + ".png"
            img_path = os.path.join(root_path, label, im_name)

            # 保存图片
            cv2.imwrite(img_path, img)

    print("done.")        

if __name__ == "__main__":
    main()
