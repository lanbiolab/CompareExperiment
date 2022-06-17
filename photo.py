'''
@File: photo.py
@Author: Dong Yiiiii
@Date: 2022/6/15 0:01
@Description:
2022/6/15
'''

import os
import shutil

photo_txt_path = r'C:\Users\DongYi\Desktop\photo.txt'
ori_photo_path = r'E:\选中照片'
ori_final_path = r'E:\选中照片\second'

with open(photo_txt_path, encoding='utf-8') as file:
    content = file.readlines()

for c in content:
    name = c.rstrip().split('\n')[0]
    photo_path = ori_photo_path + '\LBW_' + name + '.JPG'
    final_path = ori_final_path + '\LBW_' + name + '.JPG'

    shutil.copyfile(photo_path, final_path)

print(1)