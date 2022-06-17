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

if __name__=="__main__":

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # circad
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
            if (2*TP + FP + FN)==0 :
                F1 = 0
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

    print("AUC:%.4f,AUPR:%.4f"%(roc_auc, AUPR))

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/IBNPKATZ_circad_10fold_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/IBNPKATZ_circad_10fold_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("./FinalResultPng/roc-IBNPKATZ_circad_10fold.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()



