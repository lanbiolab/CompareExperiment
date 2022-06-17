'''
@Author:Dong Yi
@Date: 20201123
@Description: 这是对IBNPKATZ的denovo实现
'''


import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
import sklearn.cluster as sc
from MakeSimilarityMatrix import MakeSimilarityMatrix#计算高斯核相似性


def IBNP(circnum,disnum,circrna_disease_matrix):#应用层次聚类算法计算偏置率
    model = sc.AgglomerativeClustering(n_clusters=3)
    pred_circrna_disease_matrix = model.fit_predict(circrna_disease_matrix)#circrna的层次聚类族
    preddis_circrna_disease_matrix=model.fit_predict(np.transpose(circrna_disease_matrix))#disease的层次聚类族
    Td=[]#和疾病相关的circrna的个数
    for j in range(disnum):
        count=0
        for i in range(circnum):
            if circrna_disease_matrix[i,j]==1:
                count=count+1
        Td.append(count)
    l = 0
    m = 0
    n = 0
    for i in range(0, len(pred_circrna_disease_matrix)):#计算聚类族中每一类包含circrna的个数
        if pred_circrna_disease_matrix[i] == 0:
            l += 1
        elif pred_circrna_disease_matrix[i] == 1:
            m += 1
        elif pred_circrna_disease_matrix[i] == 2:
            n += 1
    rsum=[]#计算和疾病di相关联的circrna偏置率的和
    for j in range(disnum):
        count = 0
        for i in range(circnum):
            if pred_circrna_disease_matrix[i] == 0:
                ncr=l
            elif pred_circrna_disease_matrix[i] == 1:
                ncr=m
            elif pred_circrna_disease_matrix[i] == 2:
                ncr=n
            # 要判断分母是否为0
            if Td[j] == 0 :
                r =0
            else :
                r=ncr/Td[j]
            count+=r
        rsum.append(count)
    rxsum=[]#第一轮分配之后的cj关联的疾病偏置率的和
    for i in range(circnum):
        count = 0
        for j in range(disnum):
            if pred_circrna_disease_matrix[i] == 0:
                ncr=l
            elif pred_circrna_disease_matrix[i] == 1:
                ncr=m
            elif pred_circrna_disease_matrix[i] == 2:
                ncr=n
            if Td[j] == 0:
                r = 0
            else:
                r = ncr / Td[j]
            if rsum[j]==0:
                rx = 0
            else:
                rx= (r * Td[j]) /rsum[j]
            count+=rx
        rxsum.append(count)
    Rsum=[]#计算第二轮分配之后和疾病di有关联的circrna的偏置率的和
    for j in range(disnum):
        count = 0
        for i in range(circnum):
            if pred_circrna_disease_matrix[i] == 0:
                ncr=l
            elif pred_circrna_disease_matrix[i] == 1:
                ncr=m
            elif pred_circrna_disease_matrix[i] == 2:
                ncr=n
            if Td[j]== 0:
                r = 0
            else:
                r = ncr / Td[j]
            if rsum[j]==0:
                rx = 0
            else:
                rx= (r * Td[j]) /rsum[j]
            R=(rx*rx)/rxsum[i]
            count+=R
        Rsum.append(count)
    # Rfin=np.zeros((533,89))#初始化矩阵
    # Rfin = np.zeros((514, 62))
    # Rfin = np.zeros((312, 40))
    # Rfin = np.zeros((923, 104))
    Rfin = np.zeros((1265, 151))
    for j in range(disnum):
        count = 0
        for i in range(circnum):
            if pred_circrna_disease_matrix[i] == 0:
                ncr=l
            elif pred_circrna_disease_matrix[i] == 1:
                ncr=m
            elif pred_circrna_disease_matrix[i] == 2:
                ncr=n
            if Td[j]==0:
                r = 0
            else:
                r = ncr / Td[j]
            if rsum[j]==0:
                Rfin[i,j]=0
            else:
                Rfin[i,j]=(r*Rsum[j])/rsum[j] #最后一轮分配的值
#####################################################################################
    Tc = []
    for i in range(circnum):
        count = 0
        for j in range(disnum):
            if circrna_disease_matrix[i, j] == 1:
                count = count + 1
        Tc.append(count)
    o = 0
    p = 0
    q = 0
    for i in range(0, len(preddis_circrna_disease_matrix)):
        if preddis_circrna_disease_matrix[i] == 0:
            o += 1
        elif preddis_circrna_disease_matrix[i] == 1:
            p += 1
        elif preddis_circrna_disease_matrix[i] == 2:
            q += 1
    rdsum = []
    for j in range(circnum):
        count = 0
        for i in range(disnum):
            if preddis_circrna_disease_matrix[i] == 0:
                ndr = o
            elif preddis_circrna_disease_matrix[i] == 1:
                ndr = p
            elif preddis_circrna_disease_matrix[i] == 2:
                ndr = q
            if Tc[j]==0:
                continue
            else:
                r = ndr / Tc[j]
            count += r
        rdsum.append(count)
    rxdsum = []
    for i in range(disnum):
        count = 0
        for j in range(circnum):
            if preddis_circrna_disease_matrix[i] == 0:
                ndr = o
            elif preddis_circrna_disease_matrix[i] == 1:
                ndr = p
            elif preddis_circrna_disease_matrix[i] == 2:
                ndr = q
            if Tc[j]==0:
                continue
            else:
                r = ndr / Tc[j]
            rxd = (r * Tc[j]) / rdsum[j]
            count += rxd
        rxdsum.append(count)
    Rdsum = []
    for j in range(circnum):
        count = 0
        for i in range(disnum):
            if preddis_circrna_disease_matrix[i] == 0:
                ndr = o
            elif preddis_circrna_disease_matrix[i] == 1:
                ndr = p
            elif preddis_circrna_disease_matrix[i] == 2:
                ndr = q
            if Tc[j]==0:
                continue
            else:
                r = ndr / Tc[j]
            rxd = (r * Tc[j]) / rdsum[j]
            R = (rxd * rxd) / rxdsum[i]
            count += R
        Rsum.append(count)
    # Rfind = np.zeros((533, 89))
    # Rfind = np.zeros((514, 62))
    # Rfind = np.zeros((312, 40))
    # Rfind = np.zeros((923, 104))
    Rfind = np.zeros((1265, 151))
    for j in range(circnum):
        count = 0
        for i in range(disnum):
            if preddis_circrna_disease_matrix[i] == 0:
                ndr = o
            elif preddis_circrna_disease_matrix[i] == 1:
                ndr = p
            elif preddis_circrna_disease_matrix[i] == 2:
                ndr = q
            if Tc[j]==0:
                continue
            else:
                r = ndr / Tc[j]
            Rfind[j, i] = (r * Rsum[j]) / rdsum[j]

    SB=(Rfin+Rfind)/2
    return SB

def KATZ(SC,SD,circrna_disease_matrix):
    M1=np.hstack((SC,circrna_disease_matrix))
    M2=np.hstack((np.transpose(circrna_disease_matrix),SD))
    M=np.vstack((M1,M2))
    # I=np.eye(622,622)
    # I = np.eye(576, 576)
    # I = np.eye(352, 352)
    # I = np.eye(1027, 1027)
    I = np.eye(1416, 1416)
    S=np.linalg.inv(I-0.1*M)-I#求矩阵的逆
    # S12=S[0:533,533:622]
    # S12 = S[0:514, 514:576]
    # S12 = S[0:312, 312:352]
    # S12 = S[0:923, 923:1027]
    S12 = S[0:1265, 1265:1416]
    return S12

def GIP(circrna_disease_matrix):
    make_sim_matrix = MakeSimilarityMatrix(circrna_disease_matrix)
    return make_sim_matrix

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__=="__main__":

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('../../Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('../../Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # 需要特别考虑的六种疾病，记录在字典中
    # cancer_dict = {'glioma': 7, 'bladder cancer':9, 'breast cancer': 10,'cervical cancer': 53,'cervical carcinoma': 64,'colorectal cancer':11,'gastric cancer':19}

    # cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
    #                'colorectal cancer': 12, 'gastric cancer': 20}

    # cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
    #                'colorectal cancer': 1, 'gastric cancer': 0}

    # # circ2Traits
    # cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
    #                'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6, 'gastric cancer': 15}
    # circad
    cancer_dict = {'bladder cancer':94, 'breast cancer':53, 'triple-negative breast cancer':111, 'gliomas':56, 'glioma':76,
                    'cervical cancer':65, 'colorectal cancer':143, 'gastric cancer':28}

    # denovo start
    for i in range(115,120):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        # 计算相似高斯相似矩阵
        make_sim_matrix =GIP(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
        circnum = circrna_disease_matrix.shape[0]
        disnum = circrna_disease_matrix.shape[1]
        SC = circ_gipsim_matrix
        SD = dis_gipsim_matrix
        SB = IBNP(circnum, disnum, rel_matrix)
        # print(SB)
        SK = KATZ(SC, SD, rel_matrix)
        S = (SB + SK) / 2

        prediction_matrix = S#得到预测矩阵，这个预测矩阵中的数值为0-1之间的数
        aa = prediction_matrix.shape
        bb = roc_circrna_disease_matrix.shape
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))

        if (i % 10 == 0):
            print(prediction_matrix.shape)
            print(roc_circrna_disease_matrix.shape)

        sort_index = np.argsort(-prediction_matrix[:, i], axis=0)
        sorted_circrna_disease_row = roc_circrna_disease_matrix[:, i][sort_index]

        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []

        for cutoff in range(1, rel_matrix.shape[0] + 1):
            P_vector = sorted_circrna_disease_row[0:cutoff]
            N_vector = sorted_circrna_disease_row[cutoff:]
            TP = np.sum(P_vector == 1)
            FP = np.sum(P_vector == 0)
            TN = np.sum(N_vector == 0)
            FN = np.sum(N_vector == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            accuracy_list.append(accuracy)

        # 这里对i做一个判断，判断是否到了需要单独考虑的六种疾病
        if i in cancer_dict.values():  # 代表这一列是我们需要特别关注的一列
            top_count_list = [10, 20, 50, 100, 150, 200]
            top_count = []

            for count in top_count_list:
                p_vector = sorted_circrna_disease_row[:count]
                top_count.append(np.sum(p_vector == 1))

            # 输出这一块儿的情况
            print("列号为：" + str(i) + " 疾病名称为：" + find_key(i, cancer_dict) + " top 结果如下： \n")
            for j in range(len(top_count)):
                print("top_" + str(top_count_list[j]) + " 的结果：", top_count[j])

        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    # 用pickle数据形式将它存储下来
    with h5py.File('./IBNPKATZ_denovo_result_115_120.h5', 'w') as hf:
        hf['tpr_arr'] = tpr_arr
        hf['fpr_arr'] = fpr_arr
        hf['recall_arr'] = recall_arr
        hf['precision_arr'] = precision_arr
        hf['accuracy_arr'] = accuracy_arr
        hf['F1_arr'] = F1_arr

    # mean_denovo_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    # mean_denovo_fpr = np.mean(fpr_arr, axis=0)
    # mean_denovo_recall = np.mean(recall_arr, axis=0)
    # mean_denovo_precision = np.mean(precision_arr, axis=0)
    # mean_denovo_accuracy = np.mean(accuracy_arr, axis=0)
    # # 计算此次五折的平均评价指标数值
    # mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    # mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    # mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    # mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    # print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))
    #
    # roc_auc = np.trapz(mean_denovo_tpr, mean_denovo_fpr)
    # AUPR = np.trapz(mean_denovo_precision, mean_denovo_recall)
    # print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
    #
    # # 存储tpr，fpr,recall,precision
    # with h5py.File('./PlotFigure/IBNPKATZ_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/IBNPKATZ_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision
    #
    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean ROC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-IBNPKATZ_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()
    #
    #

