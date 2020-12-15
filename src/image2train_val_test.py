# -*- coding:utf-8 -*-
import os 
import shutil
import random


if __name__ == "__main__":
    
    root_path = "./data/cifar10"
    
    # 测试集,验证集, 训练集 占总数据的比
    test_ratio = 2
    validation_ratio = 1
    train_ratio = 7

    save_root_path = root_path + "_save"

    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    
    train_root_path = os.path.join(save_root_path, "train")
    if not os.path.exists(train_root_path):
        os.mkdir(train_root_path)

    test_root_path = os.path.join(save_root_path, "test")
    if not os.path.exists(test_root_path):
        os.mkdir(test_root_path)

    val_root_path = os.path.join(save_root_path, "val")
    if not os.path.exists(val_root_path):
        os.mkdir(val_root_path)

    for folder_name in os.listdir(root_path):
        source_path = os.path.join(root_path, folder_name)
        source_image = os.listdir(source_path)
        # 打乱顺序
        random.shuffle(source_image)
        tot_num = len(source_image)
        val_num = tot_num * (float(validation_ratio)/float(train_ratio+test_ratio+validation_ratio)) 
        test_num = tot_num * (float(validation_ratio+test_ratio)/float(train_ratio+test_ratio+validation_ratio)) 

        i = 0
        for im_name in source_image:
            print(im_name)
            i += 1
            im_file = os.path.join(source_path, im_name)
            if i < val_num:
                save_path = os.path.join(val_root_path, folder_name)
            elif val_num <= i < test_num:
                save_path = os.path.join(test_root_path, folder_name)
            else:
                save_path = os.path.join(train_root_path, folder_name)

            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            save_file = os.path.join(save_path, im_name)
            shutil.copy(im_file, save_file)

