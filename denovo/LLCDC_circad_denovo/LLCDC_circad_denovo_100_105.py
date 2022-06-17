'''
@Author: Dong Yi
@Date: 2020.11.23
@Description: 这是对文章27 LLCDC denovo的复现
'''

import random
import h5py
import math
import matplotlib.pyplot as plt
import scipy
import pickle

import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2


def compute_circ_cossim(rel_matrix):
    csc = np.zeros((rel_matrix.shape[0], rel_matrix.shape[0]))
    for i in range(csc.shape[0]):
        for j in range(csc.shape[1]):
            Vci = rel_matrix[i,:]
            Vcj = rel_matrix[j,:]
            Vci_norm = np.linalg.norm(Vci)
            Vcj_norm = np.linalg.norm(Vcj)
            # if Vci_norm==0 and Vcj_norm==0:
            #     csc[i,j] = 1
            if Vci_norm==0 or Vcj_norm==0:
                csc[i,j] = 0
            else:
                csc[i,j] = (np.dot(Vci, Vcj)) / (Vci_norm * Vcj_norm)

    return csc

def compute_dis_cossim(rel_matrix):
    csd = np.zeros((rel_matrix.shape[1], rel_matrix.shape[1]))
    for i in range(csd.shape[0]):
        for j in range(csd.shape[1]):
            Vdi = rel_matrix[:,i]
            Vdj = rel_matrix[:,j]
            multi = np.dot(Vdi, Vdj)
            if np.linalg.norm(multi) == 0:
                csd[i,j] = 0
            else:
                csd[i,j] = multi / (np.linalg.norm(multi))

    return csd


def compute_RCS(rel_matrix, circ_gipsim_matrix):
    # I = np.ones((533,1))
    # I = np.ones((514,1))
    # I = np.ones((312, 1))
    # I = np.ones((923, 1))
    I = np.ones((1265, 1))
    # W = np.zeros((533, 533))
    # W = np.zeros((514, 514))
    # W = np.zeros((312, 312))
    # W = np.zeros((923, 923))
    W = np.zeros((1265, 1265))
    for i in range(rel_matrix.shape[0]):
        xm = rel_matrix[i,:].T # xm 为 89*1
        xm = xm.reshape(-1,1)
        temp1 = rel_matrix.T - np.dot(xm, I.T)
        Cm = np.dot(temp1.T, temp1) # Cm为 533 * 533
        diagH = np.diag(circ_gipsim_matrix[i,:])
        diagH_2 = np.power(diagH, 2)
        temp2 = Cm + diagH_2
        # temp2 = np.nan_to_num(temp2) # 这里将temp2中的nan 与 inf 转为相应的数值
        w1 = np.dot(np.linalg.inv(temp2), I)
        # w1 = np.dot(scipy.linalg.pinv(temp2), I)
        # temp3 = scipy.linalg.pinv(np.dot(I.T, w1))
        temp3 = np.linalg.inv(np.dot(I.T, w1))
        w = np.dot(w1, temp3)
        w = w.flatten()
        W[:,i] = w
    W = np.maximum(W, 0)
    W = (W + W.T) / 2

    return W

def compute_RDS(rel_matrix, dis_gipsim_matrix):
    # I = np.ones((89,1))
    # I = np.ones((62, 1))
    # I = np.ones((40, 1))
    # I = np.ones((104, 1))
    I = np.ones((151, 1))
    # W1 = np.zeros((89, 89))
    # W1 = np.zeros((62, 62))
    # W1 = np.zeros((40, 40))
    # W1 = np.zeros((104, 104))
    W1 = np.zeros((151, 151))
    for i in range(rel_matrix.shape[1]):
        xd = rel_matrix[:,i] # xd是 533 * 1
        xd = xd.reshape(-1,1)
        temp1 = rel_matrix - np.dot(xd, I.T)
        Cd = np.dot(temp1.T, temp1)
        diagH = np.diag(dis_gipsim_matrix[i,:])
        diagH_2 = np.power(diagH, 2)
        temp2 = Cd + diagH_2
        # temp2 = np.nan_to_num(temp2) # 把temp2中的nan 与 inf 转化为相应的数值
        w2 = np.dot(np.linalg.inv(temp2) , I)
        # w2 = np.dot(scipy.linalg.pinv(temp2), I)
        # temp3 = scipy.linalg.pinv(np.dot(I.T, w2))
        temp3 = np.linalg.inv(np.dot(I.T, w2))
        w = np.dot(w2, temp3)
        w = w.flatten()
        W1[:,i] = w
    W1 = np.maximum(W1, 0)
    W1 = (W1 + W1.T) / 2

    return W1

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':
    # 读取关系数据
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
    # for i in range(circrna_disease_matrix.shape[1]):
    for i in range(100,105):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        # 计算相似高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        circNum = rel_matrix.shape[0]
        disNum = rel_matrix.shape[1]

        # # 这里把高斯相似矩阵存一下，方便下次直接读取
        # with h5py.File('./Data/circmi_GIP_sim_matrix.h5') as hf:
        #     hf['circ_gipsim_matrix'] = circ_gipsim_matrix
        #     hf['dis_gipsim_matrix'] = dis_gipsim_matrix

        # with h5py.File('./Data/circmi_GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # 计算circRNA disease的余弦相似性
        csc = compute_circ_cossim(rel_matrix)
        csd = compute_dis_cossim(rel_matrix)

        # 开始计算RCS 以及 RDS
        RCS = compute_RCS(rel_matrix, circ_gipsim_matrix)
        RDS = compute_RDS(rel_matrix, dis_gipsim_matrix)

        # 进行规范化
        M2 = np.sum(RCS, axis=0) # 求每一列的和，就有 533*1
        for m in range(circNum):
            for n in range(circNum):
                RCS[m, n] = RCS[m, n] / ((M2[m] * M2[n])**0.5)

        D2 = np.sum(RDS, axis=0) # 每一列的和， 89*1
        for m in range(disNum):
            for n in range(disNum):
                RDS[m,n] = RDS[m, n] /  ((D2[m] * D2[n])**0.5)

        # 接下来对circRNA的gip进行规范化，disease gip进行规范化
        M1 = np.sum(circ_gipsim_matrix, axis=0)
        for m in range(circNum):
            for n in range(circNum):
                circ_gipsim_matrix[m,n] = circ_gipsim_matrix[m, n] / ((M1[m] * M1[n])**0.5)

        D1 = np.sum(dis_gipsim_matrix, axis=0)
        for m in range(disNum):
            for n in range(disNum):
                dis_gipsim_matrix[m,n] = dis_gipsim_matrix[m ,n] / ((D1[m] * D1[n])**0.5)

        # 开始准备label propagation
        theta = 0.5
        FC = rel_matrix
        FC3 = rel_matrix
        FD = rel_matrix.T
        FD3 = rel_matrix.T
        delta = 1

        while(delta>1e-6):
            FC_ed = np.dot(theta * circ_gipsim_matrix , FC) + (1 - theta) * rel_matrix
            delta = abs(np.sum(abs(FC_ed) - abs(FC)))
            FC = FC_ed  # 在circRNA高斯相似性矩阵上传播标签，533 * 89


        delta = 1
        while(delta>1e-6):
            FRC = np.dot(theta * RCS , FC3) + (1 - theta) * rel_matrix
            delta = abs(np.sum(abs(FRC) - abs(FC3)))
            FC3 = FRC # 在RCS上传播标签， 533 * 89

        delta = 1
        while(delta>1e-6):
            FD_ed = np.dot(theta * dis_gipsim_matrix , FD) + (1 - theta) * rel_matrix.T
            delta = abs(np.sum(abs(FD_ed) - abs(FD)))
            FD = FD_ed

        delta=1
        while(delta>1e-6):
            FRD = np.dot(theta * RDS , FD3) + (1 - theta) * rel_matrix.T
            delta = abs(np.sum(abs(FRD) - abs(FD3)))
            FD3 = FRD

        F = (FC + FRC + FRD.T + FD.T) /4
        prediction_matrix = F

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
    with h5py.File('./LLCDC_denovo_result_100_105.h5', 'w') as hf:
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

    # roc_auc = np.trapz(mean_denovo_tpr, mean_denovo_fpr)
    # AUPR = np.trapz(mean_denovo_precision, mean_denovo_recall)
    # print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
    #
    # # 存储tpr，fpr,recall,precision
    # with h5py.File('./PlotFigure/LLCDC_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/LLCDC_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision

    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean denovo AUC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-LLCDC_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()


















