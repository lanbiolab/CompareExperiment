'''
@Date：2020.8.3
@Author：DongYi
@Description：针对DWNN-RLS denovo 方法进行逐步的实现
'''
import math
import random
import h5py as h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_miRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('../../Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # 需要特别考虑的六种疾病，记录在字典中
    # cancer_dict = {'glioma': 7, 'bladder cancer':9, 'breast cancer': 10,'cervical cancer': 53,'cervical carcinoma': 64,'colorectal cancer':11,'gastric cancer':19}

    # cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
    #                 'colorectal cancer': 12, 'gastric cancer': 20}

    # cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
    #                'colorectal cancer': 1, 'gastric cancer': 0}

    # circ2Traits
    cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
                   'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6, 'gastric cancer': 15}

    # # circad
    # cancer_dict = {'bladder cancer':94, 'breast cancer':53, 'triple-negative breast cancer':111, 'gliomas':56, 'glioma':76,
    #                 'cervical cancer':65, 'colorectal cancer':143, 'gastric cancer':28}

    # denovo start
    for i in range(15,20):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix
        # 计算相似高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix,make_sim_matrix.dissimmatrix
        # # 读取疾病的语义相似性
        # S_matrix_path = './Data/S.mat'
        # S_data = scio.loadmat(S_matrix_path)
        # S_matrix = S_data['S']
        # # 将语义相似性与高斯相似性进行融合
        # dis_sim_matrix = (dis_gipsim_matrix+S_matrix)/2
        dis_sim_matrix = dis_gipsim_matrix
        circ_sim_matrix = circ_gipsim_matrix

        # 根据公式初步写一下
        # 先求两个相似矩阵的特征值和特征向量
        dis_eigenvalues, dis_eigenvectors = np.linalg.eig(dis_sim_matrix)
        circ_eigenvalues, circ_eigenvectors = np.linalg.eig(circ_sim_matrix)
        circ_diag_matrix = np.diag(circ_eigenvalues)
        dis_diag_matrix = np.diag(dis_eigenvalues)

        # 首先计算向量Z
        vectorz_part1 = np.kron(circ_diag_matrix, dis_diag_matrix)
        vectorz_part2 = np.kron(circ_diag_matrix, dis_diag_matrix) + 0.2 * np.eye(vectorz_part1.shape[0])
        #求vectorz_part2的逆运算
        vectorz_part2 = np.asarray(np.asmatrix(vectorz_part2).I)

        vectorz_part3 = np.dot(np.dot(dis_eigenvectors.T, rel_matrix.T), circ_eigenvectors)
        # 把vectorz_part3 的列向量堆叠起来
        for col in range(vectorz_part3.shape[1]):
            if col==0:
                temp_vector = vectorz_part3[:,col]
            else:
                temp_vector = np.concatenate((temp_vector, vectorz_part3[:,col]))
        vectorz_part3 = temp_vector

        vectorz = np.dot(np.dot(vectorz_part1, vectorz_part2), vectorz_part3)
        # # 把vectorz还原为89*533的矩阵
        # matrix_z = np.zeros((89,533))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:,j] = vectorz[j*89:j*89+89]

        # matrix_z = np.zeros((62, 514))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:, j] = vectorz[j * 62:j * 62 + 62]

        # matrix_z = np.zeros((40, 312))
        # for j in range(matrix_z.shape[1]):
        #     matrix_z[:, j] = vectorz[j * 40:j * 40 + 40]

        matrix_z = np.zeros((rel_matrix.shape[1], rel_matrix.shape[0]))
        for j in range(matrix_z.shape[1]):
            matrix_z[:, j] = vectorz[j * rel_matrix.shape[1]:j * rel_matrix.shape[1] + rel_matrix.shape[1]]

        prediction_matrix = np.dot(np.dot(circ_eigenvectors, matrix_z.T), dis_eigenvectors.T)

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

    # 用h5py数据形式将它存储下来
    with h5py.File('./DWNN-RLS_denovo_result_15_20.h5', 'w') as hf:
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
    # with h5py.File('./PlotFigure/DWNN-RLS_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/DWNN-RLS_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision
    #
    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean ROC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-DWNN-RLS_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()







