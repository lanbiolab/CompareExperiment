'''
@Author: Dong Yi
@LastUpDateTime: 2020.10.29
@Description: 这是对 30 GBDTCDA的实现
=>由于单线程运行过于缓慢，这里使用多线程辅助运行
'''

import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix
import networkx as nx
from numpy import linalg as la
from gbdt_regressor import GradientBoostingRegressor
from multiprocessing import cpu_count, Pool
import os



# 这里原文章是将三种circRNA相似矩阵融合在一起
# 这里暂时只考虑用高斯相似性
def make_circRNA_similarity_network(circ_gipsim_matrix, rel_matrix):

    return circ_gipsim_matrix

def make_disease_similarity_network(dis_gipsim_matrix, rel_matrix):

    return dis_gipsim_matrix

def comput_graph_centrality(circ_sim_matrix, dis_sim_matrix):
    G_circ_P = nx.Graph()
    G_dis_DS = nx.Graph()

    for i in range(circ_sim_matrix.shape[0]):
        for j in range(circ_sim_matrix.shape[1]):
            if i == j:
                continue
            else:
                G_circ_P.add_weighted_edges_from([(i,j,circ_sim_matrix[i,j])])

    for i in range(dis_sim_matrix.shape[0]):
        for j in range(dis_sim_matrix.shape[1]):
            if i == j:
                continue
            else:
                if dis_sim_matrix[i,j] != 0 :
                    G_dis_DS.add_weighted_edges_from([(i,j,dis_sim_matrix[i,j])])

    circ_bet = nx.betweenness_centrality(G_circ_P)
    circ_clo = nx.closeness_centrality(G_circ_P)
    circ_eig = nx.eigenvector_centrality(G_circ_P)
    dis_bet = nx.betweenness_centrality(G_dis_DS)
    dis_clo = nx.closeness_centrality(G_dis_DS)
    dis_eig = nx.eigenvector_centrality(G_dis_DS)

    # 将上面的dict转为list
    circ_bet = [i for i in circ_bet.values()]
    circ_clo = [i for i in circ_clo.values()]
    circ_eig = [i for i in circ_eig.values()]
    dis_bet = [i for i in dis_bet.values()]
    dis_clo = [i for i in dis_clo.values()]
    dis_eig = [i for i in dis_eig.values()]

    temp4 = np.array(circ_bet + dis_bet + circ_clo + dis_clo + circ_eig + dis_eig)

    return temp4

def compute_rel_graph_centrality(rel_matrix):
    # 构建图
    G = nx.Graph()
    for i in range(rel_matrix.shape[0]):
        for j in range(rel_matrix.shape[1]):
            if rel_matrix[i, j] == 1:
                G.add_edge(i, j)
    F4_cd_bc = nx.betweenness_centrality(G)
    F4_cd_cc = nx.closeness_centrality(G)
    F4_cd_ec = nx.eigenvector_centrality(G, tol=1e-03)

    # 字典转list
    F4_cd_bc = [i for i in F4_cd_bc.values()]
    F4_cd_cc = [i for i in F4_cd_cc.values()]
    F4_cd_ec = [i for i in F4_cd_ec.values()]

    temp3 = np.array(F4_cd_bc + F4_cd_cc + F4_cd_ec)

    return temp3

def compute_unweight_matrix(circ_sim_matrix, dis_sim_matrix):
    unweight_P = np.zeros((circ_sim_matrix.shape))
    unweight_DS = np.zeros((dis_sim_matrix.shape))
    # 重新构建矩阵P DS
    for row in range(unweight_P.shape[0]):
        for col in range(unweight_P.shape[1]):
            if (circ_sim_matrix[row, col] >= np.mean(circ_sim_matrix[row, :])):
                unweight_P[row, col] = 1

    for k in range(unweight_DS.shape[0]):
        for h in range(unweight_DS.shape[1]):
            if (dis_sim_matrix[k, h] >= np.mean(dis_sim_matrix[k, :])):
                unweight_DS[k, h] = 1

    return unweight_P, unweight_DS

# 这里是构建F1特征向量
def find_feature_F1(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix):
    F1_num_nei_ci = np.sum(rel_matrix[i,:])
    F1_num_nei_dj = np.sum(rel_matrix[:,j])
    F1_sim_ave_ci = np.mean(circ_gipsim_matrix[i,:])
    F1_sim_ave_dj = np.mean(dis_gipsim_matrix[j,:])

    # 接着对相似度进行分段
    circ_low = 0
    circ_high = 0
    dis_low = 0
    dis_high=0
    for k in range(len(circ_gipsim_matrix[i])):
        if circ_gipsim_matrix[i,k] <= 0.3:
            circ_low += 1
        else:
            circ_high += 1
    for h in range(len(dis_gipsim_matrix[j])):
        if dis_gipsim_matrix[j,h] <= 0.3:
            dis_low += 1
        else:
            dis_high += 1

    F1 = np.array([F1_num_nei_ci, F1_num_nei_dj, F1_sim_ave_ci, F1_sim_ave_dj,
                        circ_low, circ_high, dis_low, dis_high])

    return F1

# 这里是构建F2特征向量
def find_feature_F2(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix, temp4, unweight_P, unweight_DS):

    F2_sum_nei_ci = np.sum(unweight_P[i, :])
    F2_sum_nei_dj = np.sum(unweight_DS[j,:])
    top10_circ_index = np.argsort(-circ_gipsim_matrix[i,:])[:10]
    top10_dis_index = np.argsort(-dis_gipsim_matrix[j,:])[:10]

    F2_K_sim_circ = circ_gipsim_matrix[i, top10_circ_index]
    F2_K_sim_dis = dis_gipsim_matrix[j , top10_dis_index]

    # 前十个与circRNA i 相似的RNA的F1值
    F2_ave_circ_list = []
    F2_ave_dis_list = []
    for index in top10_circ_index:
        f1 = find_feature_F1(index, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix)
        F2_ave_circ_list.append(f1.tolist())

    for index in top10_dis_index:
        f1_dis = find_feature_F1(i, index, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix)
        F2_ave_dis_list.append(f1_dis.tolist())

    F2_ave_feat_circ = np.mean(F2_ave_circ_list)
    F2_ave_feat_dis = np.mean(F2_ave_dis_list)

    # 接下来是用对应相似度加权的特征向量
    weighted_F2_ave_circ_list = []
    weighted_F2_ave_dis_list = []

    for index in range(len(top10_circ_index)):
        f1 = find_feature_F1(index, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix)
        f1 = np.dot(F2_K_sim_circ[index], f1)
        weighted_F2_ave_circ_list.append(f1)

    for index in range(len(top10_dis_index)):
        f1_dis = find_feature_F1(i, index, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix)
        f1_dis = np.dot(F2_K_sim_circ[index], f1_dis)
        weighted_F2_ave_dis_list.append(f1_dis)

    F2_W_ave_feat1_circ = np.mean(weighted_F2_ave_circ_list)
    F2_W_ave_feat1_dis = np.mean(weighted_F2_ave_dis_list)

    # 计算circRNA disease 的 betweenness centrality, closeness centrality , eigenvector centrality
    # 先为circRNA 与 disease 构造networkx中的图
    # G_circ_P = nx.Graph()
    # G_dis_DS = nx.Graph()
    #
    # for i in range(circ_gipsim_matrix.shape[0]):
    #     for j in range(circ_gipsim_matrix.shape[1]):
    #         if i == j:
    #             continue
    #         else:
    #             G_circ_P.add_weighted_edges_from([(i,j,circ_gipsim_matrix[i,j])])
    #
    # for i in range(dis_gipsim_matrix.shape[0]):
    #     for j in range(dis_gipsim_matrix.shape[1]):
    #         if i == j:
    #             continue
    #         else:
    #             if dis_gipsim_matrix[i,j] != 0 :
    #                 G_dis_DS.add_weighted_edges_from([(i,j,dis_gipsim_matrix[i,j])])
    #
    # circ_bet = nx.betweenness_centrality(G_circ_P)
    # circ_clo = nx.closeness_centrality(G_circ_P)
    # circ_eig = nx.eigenvector_centrality(G_circ_P)
    # dis_bet = nx.betweenness_centrality(G_dis_DS)
    # dis_clo = nx.closeness_centrality(G_dis_DS)
    # dis_eig = nx.eigenvector_centrality(G_dis_DS)
    #
    # # 将上面的dict转为list
    # circ_bet = [i for i in circ_bet.values()]
    # circ_clo = [i for i in circ_clo.values()]
    # circ_eig = [i for i in circ_eig.values()]
    # dis_bet = [i for i in dis_bet.values()]
    # dis_clo = [i for i in dis_clo.values()]
    # dis_eig = [i for i in dis_eig.values()]

    temp1 = np.array([F2_sum_nei_ci, F2_sum_nei_dj])
    temp2 = np.array(F2_K_sim_circ.tolist() + F2_K_sim_dis.tolist())
    temp3 = np.array([F2_ave_feat_circ, F2_ave_feat_dis, F2_W_ave_feat1_circ, F2_W_ave_feat1_dis])
    # temp4 = np.array(circ_bet + dis_bet + circ_clo + dis_clo + circ_eig + dis_eig)

    F2 = np.concatenate((temp1, temp2, temp3, temp4))

    return F2

def find_feature_F4(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix, temp3):
    # 首先计算关系矩阵的SVD
    U, sigma, VT = la.svd(rel_matrix)
    F4_svd_circ = U[i,:]
    F4_svd_dis = VT[j,:]

    # 计算circRNA i 的邻居数量
    F4_cd_num = np.sum(rel_matrix[i,:])
    F4_dc_num = np.sum(rel_matrix[:,j])

    # # 构建图
    # G = nx.Graph()
    # for i in range(rel_matrix.shape[0]):
    #     for j in range(rel_matrix.shape[1]):
    #         if rel_matrix[i,j] == 1:
    #             G.add_edge(i,j)
    # F4_cd_bc = nx.betweenness_centrality(G)
    # F4_cd_cc = nx.closeness_centrality(G)
    # F4_cd_ec = nx.eigenvector_centrality(G, tol=1e-03)
    #
    # # 字典转list
    # F4_cd_bc = [i for i in F4_cd_bc.values()]
    # F4_cd_cc = [i for i in F4_cd_cc.values()]
    # F4_cd_ec = [i for i in F4_cd_ec.values()]

    temp1 = np.array(F4_svd_circ.tolist() + F4_svd_dis.tolist())
    temp2 = np.array([F4_cd_num, F4_dc_num])
    # temp3 = np.array(F4_cd_bc + F4_cd_cc + F4_cd_ec)

    F4 = np.concatenate((temp1, temp2, temp3))

    return F4


if __name__ == '__main__':
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # # circ2Cancer
    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # 划分训练集为五份为后面五折实验做准备
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)
    split = math.ceil(len(one_list) / 5)

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
        # 抹除已知关系
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]

        # 计算高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        # with h5py.File('./Data/GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        circ_sim_matrix = make_circRNA_similarity_network(circ_gipsim_matrix, rel_matrix)
        dis_sim_matrix = make_disease_similarity_network(dis_gipsim_matrix, rel_matrix)

        # 由于每一折就一次计算这个circRNA 与 disease 的 betweenness centrality, closeness centrality , eigenvector centrality
        # 那么就把这个方法拿到F2计算特征向量的外面只计算一次，就不用计算那么多次了
        temp4 = comput_graph_centrality(circ_sim_matrix, dis_sim_matrix)

        # 在计算F4特征向量时也计算过rel_matrix的中间相似性这里也把它拿出来计算
        temp3 = compute_rel_graph_centrality(rel_matrix)

        # 计算F2中的unweight_P 和 unweight_DS
        unweight_P, unweight_DS = compute_unweight_matrix(circ_sim_matrix, dis_sim_matrix)

        # p = Pool(3)
        # 接下来针对相似矩阵提取特征向量
        # 原文中有四个相似矩阵，但这里缺少生物数据相似矩阵，固仅有三个特征向量
        input_X = []
        test_X = []
        input_X_label = []
        for i in range(rel_matrix.shape[0]): # rel_matrix.shape[0]
            for j in range(rel_matrix.shape[1]):
                F1 = find_feature_F1(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix)
                F2 = find_feature_F2(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix, temp4, unweight_P, unweight_DS)
                F4 = find_feature_F4(i, j, rel_matrix, circ_gipsim_matrix, dis_gipsim_matrix, temp3)
                # p.close()
                # p.join()
                temp_feature = np.concatenate((F1, F2, F4), axis=0)
                input_X.append(temp_feature)
                test_X.append(temp_feature)
                input_X_label.append(rel_matrix[i,j])

        # 加载模型
        reg = GradientBoostingRegressor()
        reg.fit(data=np.array(input_X), label=np.array(input_X_label), n_estimators=60, learning_rate=0.1, max_depth=9, min_samples_split=24)
        prediction = reg.predict(np.array(test_X))

        # 将prediction一维中的预测值转为二维矩阵
        prediction_matrix = np.zeros(rel_matrix.shape)
        for i in range(prediction_matrix.shape[0]): #prediction_matrix.shape[0]
            for j in range(prediction_matrix.shape[1]):
                prediction_matrix[i,j] = prediction[prediction_matrix.shape[1]*i+j]

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
    with h5py.File('./PlotFigure/GBDTCDA_circ2Traits_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/GBDTCDA_circ2Traits_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("./FinalResultPng/roc-GBDTCDA_circ2Traits.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()











