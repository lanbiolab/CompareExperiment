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

if __name__=="__main__":

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
    alpha=0.6
    beta=0.8
    # 5-fold start
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # 抹除已知关系，已知关系指的是A矩阵中值为一的节点
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
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
        for i in range(rel_matrix.shape[0]):  # 这样这个一维list长度就是 533 * 89 了
            for j in range(rel_matrix.shape[1]):
                circ_dis_label.append(rel_matrix[i,j])

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
        for i in range(circnum):
            itemCircRNANum = np.sum(CA[i, :])
            for j in range(circnum):
                if (CA[i, j] == 1):
                    p0Matrix[j, i] = 1.0 / itemCircRNANum

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
        for i in range(disnum):
            itemDiseaseNum = np.sum(DA[i, :])
            for j in range(disnum):
                if (DA[i, j] == 1):
                    p0Matrix[j, i] = 1.0 / itemDiseaseNum

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
        for i in range(circnum):
            for j in range(circnum):
                FSC[j]=pted1[i][j] * circ_gipsim_matrix[i][j]
            WSC[i]=FSC

        FSD = np.zeros((disnum, disnum))[:, 0]
        WSD = np.zeros((disnum, disnum))
        for i in range(disnum):
            for j in range(disnum):
                FSD[j] = pted2[i][j] * dis_gipsim_matrix[i][j]
            WSD[i] = FSD

        Circ_Dis=[]

        # for i in range(circnum):
        #     for j in range(disnum):
        #         if j==0:
        #             Circ_Dis[:,j]=(np.hstack((pted1[i],pted2[j]))).T
        #         else:
        #             Circ_Dis[:,i*j]=(np.hstack((pted1[i],pted2[j]))).T

        # 拼凑训练样本集
        for i in range(rel_matrix.shape[0]):
            for j in range(rel_matrix.shape[1]):
                circ_dis_input = []
                circ_feature = WSC[i,:]
                dis_feature = WSD[j,:]
                circ_dis_input += circ_feature.tolist()
                circ_dis_input += dis_feature.tolist()
                Circ_Dis.append(circ_dis_input)

        kn_clf = KNeighborsClassifier(n_neighbors=5)
        # kn_clf.fit(Circ_Dis, circ_dis)
        kn_clf.fit(Circ_Dis, circ_dis_label)
        kn_y = kn_clf.predict_proba(Circ_Dis)[:,1]

        cd=np.zeros((circnum,disnum))

        np.array(kn_y).reshape((circnum, disnum))
        for i in range(cd.shape[0]):
            for j in range(cd.shape[1]):
                cd[i,j] = kn_y[i*disnum + j]


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
        print(prediction_matrix.shape)
        print(roc_circrna_disease_matrix.shape)

        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix#标签矩阵等于预测矩阵加抹除关系的矩阵
        minvalue = np.min(score_matrix)#每列中的最小值
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20#？
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix,roc_circrna_disease_matrix)

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
            F1 = (2 * TP) / (2*TP + FP + FN)
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
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f"%(mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/RWR-KNN_circad_10fold_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/RWR-KNN_circad_10fold_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("./FinalResultPng/roc-RWR-KNN_circad_10fold.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()