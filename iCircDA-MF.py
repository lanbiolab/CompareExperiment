import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
import sklearn.cluster as sc
from MakeSimilarityMatrix import MakeSimilarityMatrix#计算高斯核相似性
from sklearn import preprocessing

def GIP(circrna_disease_matrix):
    make_sim_matrix = MakeSimilarityMatrix(circrna_disease_matrix)
    return make_sim_matrix

def Hdr(circnum,circ_gipsim_matrix,A):
    # a=np.zeros((533,89))
    # a = np.zeros((514, 62))
    # a = np.zeros((312, 40))
    # a = np.zeros((923, 104))
    a = np.zeros((1265, 151))
    # Ac=np.zeros((533,89))
    # Ac = np.zeros((514, 62))
    # Ac = np.zeros((312, 40))
    # Ac = np.zeros((923, 104))
    Ac = np.zeros((1265, 151))
    for i in range(circnum):
        Wc=0
        CA=a[0]
        h=(-circ_gipsim_matrix[i]).argsort()
        for j in range(2):
            Wc+=circ_gipsim_matrix[i,h[j]]
            CA+=circ_gipsim_matrix[i,h[j]]*A[h[j]]
        Ac[i]=CA/Wc
    #print(Ac)
    return Ac

def Vdr(disnum,dis_gipsim_matrix,A):

    # b = np.zeros((533, 89))
    # b = np.zeros((514, 62))
    # b = np.zeros((312, 40))
    # b = np.zeros((923, 104))
    b = np.zeros((1265, 151))
    # Ad = np.zeros((533, 89))
    # Ad = np.zeros((514, 62))
    # Ad = np.zeros((312, 40))
    # Ad = np.zeros((923, 104))
    Ad = np.zeros((1265, 151))
    for i in range(disnum):
        Wd = 0
        DA = b[:,0]
        v=(-dis_gipsim_matrix[i]).argsort()
        for j in range(2):
            Wd += dis_gipsim_matrix[i, v[j]]
            DA += dis_gipsim_matrix[i, v[j]] * A[:,v[j]]
        Ad[:,i] = DA / Wd
    #print(Ad)
    return Ad

def MaxA_Acd(A,Acd,circnum,disnum):
    # Ax=np.zeros((533,89))
    # Ax = np.zeros((514, 62))
    # Ax = np.zeros((312, 40))
    # Ax = np.zeros((923, 104))
    Ax = np.zeros((1265, 151))
    for i in range(circnum):
        for j in range(disnum):
            if A[i,j]>=Acd[i,j]:
                Ax[i,j]=A[i,j]
            else:
                Ax[i,j]=Acd[i,j]
    return Ax

def Diagonal_matrix(circ_gipsim_matrix,dis_gipsim_matrix,circnum,disnum):
    # Ic=np.zeros((533,533))
    # Ic = np.zeros((514, 514))
    # Ic = np.zeros((312, 312))
    # Ic = np.zeros((923, 923))
    Ic = np.zeros((1265, 1265))
    # Id=np.zeros((89,89))
    # Id = np.zeros((62, 62))
    # Id = np.zeros((40, 40))
    # Id = np.zeros((104, 104))
    Id = np.zeros((151, 151))
    for i in range(circnum):
        ci=0
        for j in range(circnum):
            ci+=circ_gipsim_matrix[i,j]
        Ic[i,i]=ci
    for p in range(disnum):
        di=0
        for q in range(disnum):
            di+=dis_gipsim_matrix[p,q]
        Id[p,p]=di
    return Ic,Id

def Cx_and_Dx(Ax,circ_gipsim_matrix,dis_gipsim_matrix,C,D,Ic,Id,alpha,beta,circnum,disnum,r):
    X=np.dot(Ax,D)+beta*np.dot(circ_gipsim_matrix,C)
    Y=(alpha+1)*np.dot(np.dot(C,D.T),D)+beta*np.dot(Ic,C)
    # Cx=np.zeros((533,70))
    # Cx = np.zeros((514, 70))
    # Cx = np.zeros((312, 70))
    # Cx = np.zeros((923, 70))
    Cx = np.zeros((1265, 70))
    for i in range(circnum):
        for j in range(r):
            Cx[i][j]=X[i][j]/Y[i][j]
    P=np.dot(Ax.T,C)+beta*np.dot(dis_gipsim_matrix,D)
    Q=(alpha+1)*np.dot(np.dot(D,C.T),C)+beta*np.dot(Id,D)
    # Dx=np.zeros((89,70))
    # Dx = np.zeros((62, 70))
    # Dx = np.zeros((40, 70))
    # Dx = np.zeros((104, 70))
    Dx = np.zeros((151, 70))
    for i in range(disnum):
        for j in range(r):
            if Q[i][j]==0:
                Dx[i][j]=0
            else:
                Dx[i][j]=P[i][j]/Q[i][j]
            #if Dx[i][j]>=1:
               # Dx[i][j]=1
    return Cx,Dx

def Matrix_factorzation(Ax, C,D,Cx, Dx,Ic,Id, r,alpha, beta, steps=1):
    D = D.T
    Dx=Dx.T
    result = []
    for step in range(steps):
        for i in range(len(Ax)):
            for j in range(len(Ax[i])):
                if Ax[i][j] > 0:
                    eij = Ax[i][j] - np.dot(C[i, :], D[:, j])
                    for k in range(r):
                        C[i][k]=C[i][k]*Cx[i][k]
                        D[k][j]=D[k][j]*Dx[k][j]
        eR = np.dot(C, D)
        e = 0
        for i in range(len(Ax)):
            for j in range(len(Ax[i])):
                if Ax[i][j] > 0:
                    e = e + pow(Ax[i][j] - np.dot(C[i, :], D[:, j]), 2)
                    for k in range(r):
                        e = e + alpha * C[i][k]*D[k][j]
        e+=beta*(np.trace(np.dot(np.dot(C.T,(Ic-circ_gipsim_matrix)),C))+np.trace(np.dot(np.dot(D,(Id-dis_gipsim_matrix)),D.T)))
        result.append(e)
        if e < 0.001:
            break
    return C, D.T, result
    
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
    alpha=0.002
    beta=0.001
    r=70
    k=2
    # 5-fold start
    for i in range(0, len(one_list), split):
        # C = np.random.rand(533, r)
        # C = np.random.rand(514, r)
        # C = np.random.rand(312, r)
        C = np.random.rand(circrna_disease_matrix.shape[0], r)
        # D = np.random.rand(89, r)
        # D = np.random.rand(62, r)
        # D = np.random.rand(40, r)
        D = np.random.rand(circrna_disease_matrix.shape[1], r)
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
        #with h5py.File('./Data/Gip_sim_matrix.h5')as hf:
            #hf['circ_gipsim_matrix']=circ_gipsim_matrix
            #hf['dis_gipsim_matrix']=dis_gipsim_matrix
        #print('aaaaaaaaaaaaaaaaaa')
        #with h5py.File('./Data/Gip_sim_matrix.h5','r')as hf:
            #circ_gipsim_matrix=hf['circ_gipsim_matrix'][:]
            #dis_gipsim_matrix=hf['dis_gipsim_matrix'][:]
        circnum = circrna_disease_matrix.shape[0]
        disnum = circrna_disease_matrix.shape[1]
        Ac=Hdr(circnum,circ_gipsim_matrix,rel_matrix)
        Ad=Vdr(disnum,dis_gipsim_matrix,rel_matrix)
        Acd=(Ac+Ad)/2
        Ax=MaxA_Acd(rel_matrix,Acd,circnum,disnum)
        Ic,Id=Diagonal_matrix(circ_gipsim_matrix,dis_gipsim_matrix,circnum,disnum)
        Cx,Dx=Cx_and_Dx(Ax,circ_gipsim_matrix,dis_gipsim_matrix,C,D,Ic,Id,alpha,beta,circnum,disnum,r)
        C,D,result=Matrix_factorzation(Ax, C,D,Cx, Dx,Ic,Id, r,  alpha, beta,steps=1)
        A=np.dot(C,D.T)
        # print(A)
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i,j]>1 or A[i][j]<0:
                    A[i,j]==0

        prediction_matrix = A#得到预测矩阵，这个预测矩阵中的数值为0-1之间的数
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
    with h5py.File('./PlotFigure/iCircDA_MF_circad_10fold_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/iCircDA_MF_circad_10fold_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("./FinalResultPng/roc-iCircDA_MF_circad_10fold.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()