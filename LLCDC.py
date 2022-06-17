'''
@Author: Dong Yi
@Date: 2020.10.9
@Description: 这是对文章27 LLCDC的复现
'''

import random
import h5py
import math
import matplotlib.pyplot as plt
import scipy

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
    I = np.ones((rel_matrix.shape[0],1))
    # W = np.zeros((533, 533))
    # W = np.zeros((514, 514))
    # W = np.zeros((312, 312))
    W = np.zeros((rel_matrix.shape[0], rel_matrix.shape[0]))
    for i in range(rel_matrix.shape[0]):
        xm = rel_matrix[i,:].T # xm 为 89*1
        xm = xm.reshape(-1,1)
        temp1 = rel_matrix.T - np.dot(xm, I.T)
        Cm = np.dot(temp1.T, temp1) # Cm为 533 * 533
        diagH = np.diag(circ_gipsim_matrix[i,:])
        diagH_2 = np.power(diagH, 2)
        temp2 = Cm + diagH_2
        # temp2 = np.nan_to_num(temp2) # 这里将temp2中的nan 与 inf 转为相应的数值
        # 判断temp2 是否有nan 或者 inf 值
        nan_array = np.argwhere(np.isnan(temp2))
        inf_array = np.argwhere(np.isinf(temp2))
        for m in range(len(nan_array)):
            nan_index = nan_array[m]
            temp2[nan_index[0], nan_index[1]]=0
        for n in range(len(inf_array)):
            inf_index = inf_array[n]
            temp2[inf_index[0], inf_index[1]] = 1
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
    I = np.ones((rel_matrix.shape[1], 1))
    # W1 = np.zeros((89, 89))
    # W1 = np.zeros((62, 62))
    # W1 = np.zeros((40, 40))
    W1 = np.zeros((rel_matrix.shape[1], rel_matrix.shape[1]))
    for i in range(rel_matrix.shape[1]):
        xd = rel_matrix[:,i] # xd是 533 * 1
        xd = xd.reshape(-1,1)
        temp1 = rel_matrix - np.dot(xd, I.T)
        Cd = np.dot(temp1.T, temp1)
        diagH = np.diag(dis_gipsim_matrix[i,:])
        diagH_2 = np.power(diagH, 2)
        temp2 = Cd + diagH_2
        temp2 = np.nan_to_num(temp2) # 把temp2中的nan 与 inf 转化为相应的数值
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


if __name__ == '__main__':
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # 划分训练集为五份为后面五折实验做准备
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)
    split = math.ceil(len(one_list) / 10)

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # 抹除已知关系
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        # 计算相似高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        circNum = rel_matrix.shape[0]
        disNum = rel_matrix.shape[1]

        # # 这里把高斯相似矩阵存一下，方便下次直接读取
        # with h5py.File('./Data/circ_GIP_sim_matrix.h5') as hf:
        #     hf['circ_gipsim_matrix'] = circ_gipsim_matrix
        #     hf['dis_gipsim_matrix'] = dis_gipsim_matrix

        # with h5py.File('./Data/circ_GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # # 计算circRNA disease的余弦相似性
        # csc = compute_circ_cossim(rel_matrix)
        # csd = compute_dis_cossim(rel_matrix)

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
        print(prediction_matrix.shape)
        print(roc_circrna_disease_matrix.shape)

        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix
        minvalue = np.min(score_matrix)
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix,
                                                                                   roc_circrna_disease_matrix)

        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
            P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
            N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
            TP = np.sum(P_matrix == 1)
            FP = np.sum(P_matrix == 0)
            TN = np.sum(N_matrix == 0)
            FN = np.sum(N_matrix == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            accuracy_list.append(accuracy)

        # # 下面是对top50，top100，top200的预测准确的计数
        # top_list = [50, 100, 200]
        # for num in top_list:
        #     P_matrix = sorted_circrna_disease_matrix[0:num, :]
        #     N_matrix = sorted_circrna_disease_matrix[num:sorted_circrna_disease_matrix.shape[0] + 1, :]
        #     top_count = np.sum(P_matrix == 1)
        #     print("top" + str(num) + ": " + str(top_count))

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

    mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
    # 计算此次五折的平均评价指标数值
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/LLCDC_circad_10fold_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/LLCDC_circad_10fold_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean AUC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("./FinalResultPng/roc-LLCDC_circad_10fold.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()


















