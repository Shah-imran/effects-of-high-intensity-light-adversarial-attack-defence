# import OS module
import os
import random
import shutil
import sys

# Get the list of all files and directories
train_path = os.path.join(os.getcwd(), 'train')
train_img_path = train_path + "/images"
train_label_path = train_path + "/labels"
val_path = os.path.join(os.getcwd(), 'val')
val_img_path = val_path + "/images"
val_label_path = val_path + "/labels"
dir_list = os.listdir(train_img_path)

percentage = float(9)
k = len(dir_list) * percentage // 100

sample_list = random.sample(dir_list, int(k))

for i in sample_list:
    try:
        file_name = i.split('.jpg')[0]
    except:
        file_name = i.split('.jpeg')[0]
    try:
        label_name = file_name + '.txt'
        shutil.move(train_img_path + '/' + i, val_img_path + '/' + i)
        shutil.move(train_label_path + '/' + label_name, val_label_path + '/' + label_name)
    except:
        print('Transfer failed for:')
        print("mv " + train_img_path + '/' + i + ' ' + val_img_path + '/')
        print("mv " + train_label_path + '/' + label_name + ' ' + val_label_path + '/')
