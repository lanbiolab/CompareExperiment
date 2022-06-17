'''
@Author : Dong Yi, Zhang Hongyu
@Date:20201104
@Description: 对学弟编写的代码进行了一定的修改
'''

import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.setrecursionlimit(100000)

def GIP(circrna_disease_matrix):
    make_sim_matrix = MakeSimilarityMatrix(circrna_disease_matrix)
    return make_sim_matrix

def CA_DA(SC,SD,circnum,disnum,alpha,beta):
    CA=np.zeros((circnum,circnum))
    DA=np.zeros((disnum,disnum))
    for i in range(circnum):
        for j in range(circnum):
            if SC[i][j]>=alpha:
                CA[i][j]=1
            else:
                CA[i][j]=0
    for i in range(disnum):
        for j in range(disnum):
            if SD[i][j]>=beta:
                DA[i][j]=1
            else:
                DA[i][j]=0
    return CA,DA

def p(WMatrix, t, p0Matrix, r=0.7):
    if(t == 0):
        return p0Matrix
    else:
        return (1-r) * np.matmul(WMatrix, p(WMatrix,t-1,p0Matrix)) + r*p0Matrix

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
    alpha=0.6
    beta=0.8

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
    for i in range(105,110):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]

        # 计算相似高斯相似矩阵
        make_sim_matrix =GIP(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        # with h5py.File('./Data/GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        circnum = circrna_disease_matrix.shape[0]
        disnum = circrna_disease_matrix.shape[1]

        CA,DA=CA_DA(circ_gipsim_matrix,dis_gipsim_matrix,circnum,disnum,alpha,beta)

        # circ_dis=np.zeros((533 + 89, 533 * 89))

        # 这个circ_dis_label 是为了存储关系矩阵中类别的label的
        circ_dis_label = []
        for m in range(rel_matrix.shape[0]):  # 这样这个一维list长度就是 533 * 89 了
            for n in range(rel_matrix.shape[1]):
                circ_dis_label.append(rel_matrix[m,n])

        # '''
        # 下面这段代码再仔细看看
        # '''
        # for i in range(circnum):
        #     for j in range(disnum):
        #         if j == 0:
        #             circ_dis[:, j] = (np.hstack((CA[i], DA[j]))).T
        #         else:
        #             circ_dis[:, i * j] = (np.hstack((CA[i], DA[j]))).T

        p0Matrix = np.zeros((circnum, circnum))
        for m in range(circnum):
            itemCircRNANum = np.sum(CA[i, :])
            for n in range(circnum):
                if (CA[m, n] == 1):
                    p0Matrix[n, m] = 1.0 / itemCircRNANum

        t = 1
        pt = p0Matrix
        CircRNANum = circnum * circnum

        # 对CA， DA进行规范化
        CA_norm = CA / np.linalg.norm(CA)
        DA_norm = DA / np.linalg.norm(DA)

        # 两个RWR循环迭代
        while (True):
            pted1 = p(CA_norm, t, p0Matrix)
            Delta = abs(pted1 - pt)
            if (np.sum(Delta) / CircRNANum < 1e-6):  # 重启随机游走停止的限制条件
                break
            pt = pted1
            t += 1

        p0Matrix = np.zeros((disnum, disnum))
        for m in range(disnum):
            itemDiseaseNum = np.sum(DA[i, :])
            for n in range(disnum):
                if (DA[m, n] == 1):
                    p0Matrix[n, m] = 1.0 / itemDiseaseNum

        t = 1
        pt = p0Matrix
        DiseaseNum = disnum * disnum

        while (True):
            pted2 = p(DA_norm, t, p0Matrix)
            Delta = abs(pted2 - pt)
            if (np.sum(Delta) / DiseaseNum < 1e-6):  # 重启随机游走停止的限制条件
                break
            pt = pted2
            t += 1
        FSC=np.zeros((circnum,circnum))[:,0]
        WSC=np.zeros((circnum,circnum))
        for m in range(circnum):
            for n in range(circnum):
                FSC[n]=pted1[m][n] * circ_gipsim_matrix[m][n]
            WSC[m]=FSC

        FSD = np.zeros((disnum, disnum))[:, 0]
        WSD = np.zeros((disnum, disnum))
        for m in range(disnum):
            for n in range(disnum):
                FSD[n] = pted2[m][n] * dis_gipsim_matrix[m][n]
            WSD[m] = FSD

        Circ_Dis=[]

        # for i in range(circnum):
        #     for j in range(disnum):
        #         if j==0:
        #             Circ_Dis[:,j]=(np.hstack((pted1[i],pted2[j]))).T
        #         else:
        #             Circ_Dis[:,i*j]=(np.hstack((pted1[i],pted2[j]))).T

        # 拼凑训练样本集
        for m in range(rel_matrix.shape[0]):
            for n in range(rel_matrix.shape[1]):
                circ_dis_input = []
                circ_feature = WSC[m,:]
                dis_feature = WSD[n,:]
                circ_dis_input += circ_feature.tolist()
                circ_dis_input += dis_feature.tolist()
                Circ_Dis.append(circ_dis_input)

        kn_clf = KNeighborsClassifier(n_neighbors=5)
        # kn_clf.fit(Circ_Dis, circ_dis)
        kn_clf.fit(Circ_Dis, circ_dis_label)
        kn_y = kn_clf.predict_proba(Circ_Dis)[:,1]

        cd=np.zeros((circnum,disnum))

        np.array(kn_y).reshape((circnum, disnum))
        for m in range(cd.shape[0]):
            for n in range(cd.shape[1]):
                cd[m,n] = kn_y[m*disnum + n]


        # for k in range(disnum*circnum):
        #     i = k / circnum
        #     j = k % disnum
        #     if kn_y[:,k]==Circ_Dis[:,k]:
        #         cd[i][j]=1
        #     else:
        #         cd[i][j]=0

        prediction_matrix = cd#得到预测矩阵，这个预测矩阵中的数值为0-1之间的数
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
    with h5py.File('./RWRKNN_denovo_result_105_110.h5', 'w') as hf:
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
    # with h5py.File('./PlotFigure/RWR-KNN_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/RWR-KNN_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision
    #
    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean denovo ROC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-RWR-KNN_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()