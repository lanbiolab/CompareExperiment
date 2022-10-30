import numpy as np
import pandas as pd

def get_miRNA_cancer_data():
    cancer_list = (pd.read_excel('disease_list.xlsx')).values.tolist()
    cancer_list_ = []
    for i in range(len(cancer_list)):
        str = cancer_list[i][0].lower()
        cancer_list_.append(str)
    miRNA_list = []
    miRNA_cancer_list = []
    with open('miRNA-cancer.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去除文本中的换行符
            line = line.strip().split('\t')
            line1 = line[0].lower()
            line2 = line[1].lower()
            if line2 in cancer_list_:
                miRNA_cancer_list.append([line1, line2])
                miRNA_list.append(line1)
    with open('miRNA_cancer.txt', 'w+', encoding='utf-8') as f:
        for l in miRNA_cancer_list:
            f.write(l[0] + "\t" + "2" + "\t" + l[1]+ "\n")
    miRNA_list = list(set(miRNA_list))
    with open('miRNA_list.txt', 'w+', encoding='utf-8') as f:
        for miRNA in miRNA_list:
            f.write(miRNA+'\n')
    #             miRNA_cancer_list.append([line1, line2])
    # with open('miRNA-cancer.csv', 'w+', encoding='utf-8') as f:
    #     for miRNA_cancer in miRNA_cancer_list:
    #         f.write(miRNA_cancer[0]+"\t"+miRNA_cancer)

def get_lncRNA_cancer_data():
    lncRNA_list = (pd.read_excel('lncRNA-cancer.xlsx', usecols=[0])).values.tolist()
    cancer_list = (pd.read_excel('lncRNA-cancer.xlsx', usecols=[1])).values.tolist()
    cancer_list_standard = (pd.read_excel('disease_list.xlsx')).values.tolist()
    lncRNA_list_ = []
    lncRNA_cancer_list = []
    for i in range(len(lncRNA_list)):
        if cancer_list[i] in cancer_list_standard:
            lncRNA_list_.append(lncRNA_list[i][0].lower())
            lncRNA_cancer_list.append([lncRNA_list[i][0].lower(), cancer_list[i][0].lower()])
    with open('lncRNA_cancer.txt', 'w+', encoding='utf-8') as f:
        for l in lncRNA_cancer_list:
            f.write(l[0] + '\t' + '3' + '\t' + l[1] + '\n')

    lncRNA_list_ = list(set(lncRNA_list_))
    with open('lncRNA_list.txt', 'w+', encoding='utf-8') as f:
        for lncRNA in lncRNA_list_:
            f.write(lncRNA+'\n')

def get_circRNA_miRNA_data():
    circRNA_list = (pd.read_excel('circRNA_list.xlsx')).values.tolist()
    for i in range(len(circRNA_list)):
        circRNA_list[i] = circRNA_list[i][0].lower()
    miRNA_list = []
    circRNA_miRNA_list = []
    with open('miRNA_list.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            miRNA_list.append(line)
    with open('circRNA-miRNA.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去除文本中的换行符
            line = line.strip().split('\t')
            line1 = line[0].lower()
            line2 = line[1].lower()
            if line2 in circRNA_list and line1 in miRNA_list:
                circRNA_miRNA_list.append([line1, line2])
    with open('circRNA_miRNA.txt', 'w+', encoding='utf-8') as f:
        for l in circRNA_miRNA_list:
            f.write(l[0]+'\t'+'4'+'\t'+l[1]+'\n')

def get_circRNA_cancer_data():
    data = pd.read_csv('circRNA_cancer.csv', delimiter=',')
    cancer_list = list(data)
    cancer_list = cancer_list[1:]
    circRNA_list = data.values[:, 0]
    matrix = data.values[1:515, 1:63]
    with open('circRNA_cancer.txt', 'w+', encoding='utf-8') as f:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]>0:
                    f.write(circRNA_list[i]+'\t'+'1'+'\t'+cancer_list[j]+'\n')
    # print(type(matrix))
    with open('circRNA.txt', 'w+', encoding='utf-8') as f:
        for circRNA in circRNA_list:
            f.write(circRNA+'\n')
    with open('cancer.txt', 'w+', encoding='utf-8') as f:
        for cancer in cancer_list:
            f.write(cancer+'\n')

if __name__ =="__main__":
    # get_miRNA_cancer_data()
    # get_lncRNA_cancer_data()
    # get_circRNA_miRNA_data()
    get_circRNA_cancer_data()