'''
@Author: Dong Yi
@Description: 这是一段对GBDTCDA多线程的改良
2020.12.15 这是对多线程改写而成的denovo
2020.12.16 GBDT用这个做denovo
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
import multiprocessing
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

# 这里定义子子线程处理每个批次的数据的部分
def batch_process(start, end, rel_matrix, circ_sim_matrix, dis_sim_matrix):
    # 这里担心超过rel_matrix 的横坐标范围
    if(end > rel_matrix.shape[0]):
        end = rel_matrix.shape[0]

    input_X = []
    test_X = []
    input_X_label = []

    # 那么就把这个方法拿到F2计算特征向量的外面只计算一次，就不用计算那么多次了
    temp4 = comput_graph_centrality(circ_sim_matrix, dis_sim_matrix)

    # 在计算F4特征向量时也计算过rel_matrix的中间相似性这里也把它拿出来计算
    temp3 = compute_rel_graph_centrality(rel_matrix)

    # 计算F2中的unweight_P 和 unweight_DS
    unweight_P, unweight_DS = compute_unweight_matrix(circ_sim_matrix, dis_sim_matrix)

    for i in range(start,end):
        for j in range(rel_matrix.shape[1]):
            F1 = find_feature_F1(i, j, rel_matrix, circ_sim_matrix, dis_sim_matrix)
            F2 = find_feature_F2(i, j, rel_matrix, circ_sim_matrix, dis_sim_matrix, temp4, unweight_P, unweight_DS)
            F4 = find_feature_F4(i, j, rel_matrix, circ_sim_matrix, dis_sim_matrix, temp3)

            temp_feature = np.concatenate((F1, F2, F4), axis=0)
            input_X.append(temp_feature.tolist())
            test_X.append(temp_feature.tolist())
            input_X_label.append(rel_matrix[i, j])

    return input_X, test_X, input_X_label

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

    temp1 = np.array([F2_sum_nei_ci, F2_sum_nei_dj])
    temp2 = np.array(F2_K_sim_circ.tolist() + F2_K_sim_dis.tolist())
    temp3 = np.array([F2_ave_feat_circ, F2_ave_feat_dis, F2_W_ave_feat1_circ, F2_W_ave_feat1_dis])

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

    temp1 = np.array(F4_svd_circ.tolist() + F4_svd_dis.tolist())
    temp2 = np.array([F4_cd_num, F4_dc_num])

    F4 = np.concatenate((temp1, temp2, temp3))

    return F4

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('../../Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('../../Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # 划分训练集为五份为后面五折实验做准备
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)

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
    #                 'colorectal cancer': 1, 'gastric cancer': 0}

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


        # 分批次处理
        input_X = []
        test_X = []
        input_X_label = []

        batch_split = math.ceil(rel_matrix.shape[0] / 10)

        results = []
        pool = multiprocessing.Pool(processes=10)
        for k in range(10):
            results.append(pool.apply_async(batch_process,
                                            args=(k * batch_split, k * batch_split + batch_split, rel_matrix, circ_sim_matrix, dis_sim_matrix)))

        pool.close()
        pool.join()

        h = 1
        for res in results:
            locals()['batch_' + str(h) + '_input_X'], locals()['batch_' + str(h) + '_test_X'], locals()['batch_' + str(h) + '_input_X_label'] = res.get()
            h += 1


        for h in range(10):
            input_X += locals()['batch_' + str(h + 1) + '_input_X']
            test_X += locals()['batch_' + str(h + 1) + '_test_X']
            input_X_label += locals()['batch_' + str(h + 1) + '_input_X_label']

        # 加载模型
        reg = GradientBoostingRegressor()
        reg.fit(data=np.array(input_X), label=np.array(input_X_label), n_estimators=60, learning_rate=0.1, max_depth=9,
                min_samples_split=24)
        prediction = reg.predict(np.array(test_X))

        # 将prediction一维中的预测值转为二维矩阵
        prediction_matrix = np.zeros(rel_matrix.shape)
        for m in range(prediction_matrix.shape[0]):
            for n in range(prediction_matrix.shape[1]):
                prediction_matrix[m, n] = prediction[rel_matrix.shape[1] * m + n]

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
    with h5py.File('./GBDT_denovo_result_115_120.h5', 'w') as hf:
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
    # with h5py.File('./PlotFigure/GBDT_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/GBDT_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision
    #
    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean denovo AUC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-GBDT_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()










