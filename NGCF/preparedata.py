'''
@Description:这里是为了整理circRNA-disease的数据并分为
train.txt 和 test。txt
@Date：
'''
import math
import random

import h5py as h5py
import numpy as np


#=======================================================================================================================================#
'''
这一段是专门为case-study代码提取的关系
将所有已知关系对全部都作为训练集
测试集与训练集有重复的位置
'''
with h5py.File('../Data/circRNA-disease/disease-circRNA.h5', 'r') as hf:
    circrna_disease_matrix = hf['infor'][:]
    circrna_disease_matrix_val = circrna_disease_matrix.copy()
index_tuple = (np.where(circrna_disease_matrix == 1))
one_list = list(zip(index_tuple[0], index_tuple[1]))
# random.shuffle(one_list)
split = math.ceil(len(one_list) / 5)

for (u,i) in one_list:
    with open("../Data/case-study/train.txt","a") as f:
        f.write(str(u) + " " + str(i) + "\n")
        f.close()
test_index = one_list[0:0 + split]
for (u,i) in test_index:
    with open("../Data/case-study/test.txt","a") as h:
        h.write(str(u) + " " + str(i) + "\n")
        h.close()
#======================================================================================================================================================#
'''
下面这部分是五折交叉实验中
分别对五折的每一折数据进行处理
'''

# # 从one_list_file中读取数据
# with h5py.File('./one_list_file/train_test_fold5.h5', 'r') as f:
#     test_index = f['test_index'][:]
#     train_index = f['train_index'][:]
# # 将test_index 和 train_index 中的列表转化为元组
# test_temp=[]
# train_temp=[]
# for test in test_index:
#     test_temp.append(tuple(test))
# for train in train_index:
#     train_temp.append(tuple(train))
# test_index = test_temp
# train_index = train_temp
#
# for row in range(circrna_disease_matrix.shape[0]):
#     for col in range(circrna_disease_matrix.shape[1]):
#         with open("../Data/circRNA-disease-fold5/train.txt","a") as f:
#             if (row,col) in train_index:
#                 f.write(str(row)+" "+str(col)+"\n")
#                 f.close()
#         with open("../Data/circRNA-disease-fold5/test.txt","a") as h:
#             if (row,col) in test_index:
#                 h.write(str(row) + " " +str(col)+"\n")
#                 h.close()
# print("finished")