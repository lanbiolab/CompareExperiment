'''
@Author: Dong Yi
@Date:2020.12.9
@Description: 这是对新的数据库的数据进行整理的程序
主要是对circ2Traits 数据库整理
最终整理为circ_list dis_list 以及 association_matrix
'''

# 读取txt文件
import csv

import h5py
import numpy as np

circ_list = []
dis_list = []

with open('./Data/circ2Traits/all_disease_set.txt','r') as f:
    while(True):
        line = f.readline()
        if not line:
            break

        _, _, NM, circ_temp, dis_temp, _ = line.split('\t')
        # 接着对circ_temp进一步处理
        circ_temp_list = circ_temp.split(',')
        circ_temp_list = [i.lower() for i in circ_temp_list if i!= '']
        # 这一步首先就对circ_temp_list去重
        circ_temp_list = list(set(circ_temp_list))
        circ_list += circ_temp_list

        # 接着进一步对dis_temp进一步处理
        dis_temp_list = dis_temp.split(',')
        dis_name = dis_temp_list[0]
        dis_name_temp_list = dis_name.split('(')
        dis_name = dis_name_temp_list[0]
        dis_name = (dis_name.strip()).lower()
        dis_list.append(dis_name)

        # 将circRNA关系与disease关系计入csv文件中
        for i in circ_temp_list:
            with open('./Data/circ2Traits/circRNA_disease.csv', 'a') as hs:
                csv_writer = csv.writer(hs)
                csv_writer.writerow([str(i), dis_name])

# 对list去重
circ_list = list(set(circ_list))
dis_list = list(set(dis_list))

# 创建关系矩阵
association_matrix = np.zeros((len(circ_list), len(dis_list)))

with open('./Data/circ2Traits/circRNA_disease.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        circname = row[0]
        disname = row[1]
        circ_index = circ_list.index(circname)
        dis_index = dis_list.index(disname)
        association_matrix[circ_index, dis_index] = 1

# 记录数据
# 存储circ_list, dis_list, association_matrix
for k in circ_list:
    with open('./Data/circ2Traits/circRNA_list.csv', 'a') as hd:
        csv_writer = csv.writer(hd)
        csv_writer.writerow([k])

for d in dis_list:
    with open('./Data/circ2Traits/disease_list.csv', 'a') as hd:
        csv_writer = csv.writer(hd)
        csv_writer.writerow([d])

with h5py.File('./Data/circ2Traits/circRNA_disease.h5') as hf:
    hf['infor'] = association_matrix





