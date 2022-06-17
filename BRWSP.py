'''
@Author: Dong Yi
@Date: 202.9.27
@Description: 这是文章25 BRWSP的复现
'''

import math
import random
import h5py
import numpy as np
import sortscore
import matplotlib.pyplot as plt
from MakeSimilarityMatrix import MakeSimilarityMatrix

def CS(relmatrix, circ_gipsim_matrix):

    return  circ_gipsim_matrix

def DS(relmatrix, dis_gipsim_matrix):

    return dis_gipsim_matrix

# 计算当前节点邻居节点的可能性的大小
def compute_neighbor_prop(current_neighborhood_list, prev_neighborhood_list, current_node, prev_node, NMH, temp_path):

    prop_list = []
    for i in range(len(current_neighborhood_list)):
        x_node = current_neighborhood_list[i]
        # 首先判断参数
        if (x_node in current_neighborhood_list) and (x_node in prev_neighborhood_list):
            pesai = 0.12 # q为0.12
        else:
            pesai = 1 - 0.12
        if (x_node in current_neighborhood_list) and (x_node not in temp_path):
            sum_NMH = 0
            # 这个地方是把邻居节点current_neighborhood_list 中的节点的NMH中的值都加上
            for j in range(len(current_neighborhood_list)):
                sum_NMH += NMH[current_node, current_neighborhood_list[j]]
            propablity = (pesai * NMH[current_node, x_node]) / sum_NMH
            prop_list.append(propablity)
        else:
            prop_list.append(0)

    return prop_list


# 找到当前节点的邻居节点是哪些
def find_neighborhood(current_node, H_matrix):

    # Hmatrix中current_node这一横行都是当前节点的邻居节点
    neighborhood_array = np.where(H_matrix[current_node,:]!=0) # 从长度是622该节点的向量里寻找不为0的下标
    neighborhood_list = []
    for i in range(len(neighborhood_array[0])):
        neighborhood_list.append(neighborhood_array[0][i])

    return neighborhood_list

# 找到起点到终点的所有路径
def find_path(start, destination, L, H_matrix, NMH, maxiter):
    total_path_list = []
    indexed_set = set()
    for i in range(maxiter):
        current_node = start
        prev_node = start
        temp_path = []
        # 在路径长度限定范围内
        for j in range(L):
            temp_path.append(current_node)
            # 判断当前节点是不是最终节点
            if current_node == destination:
                break;
            current_neighborhood_list = find_neighborhood(current_node, H_matrix)
            prev_neighborhood_list = find_neighborhood(prev_node, H_matrix)
            if j+1 == 1: # 如果k为1
                next_node = current_neighborhood_list[-1]
                prev_node = current_node
                current_node = next_node
            else: # 如果k不为1
                prop_list = compute_neighbor_prop(current_neighborhood_list, prev_neighborhood_list, current_node, prev_node, NMH, temp_path)
                max_propablity_index = prop_list.index(max(prop_list))
                while(max_propablity_index in indexed_set):
                    prop_list[max_propablity_index] = 0 # 如果这个最大值已经找到过，则将这个list对应位置赋0
                    max_propablity_index = prop_list.index(max(prop_list))
                indexed_set.add(max_propablity_index)
                next_node = current_neighborhood_list[max_propablity_index]
                prev_node = current_node
                current_node = next_node
            if j == L-1: # 如果已经走到了最后一个节点
                temp_path.append(current_node)
        if temp_path[-1] != destination:
            continue
        else: # 路径最后一个结点是我的终点
            total_path_list.append(temp_path)
    # 对total_path_list去重
    total_path_list = list(set(total_path_list))

    return total_path_list


# 从这里开始预测从起点i 到 终点 j的预测值
def predict_sore(start, destination, L, H_matrix, NMH, maxiter):

    total_path_list = find_path(start, destination, L, H_matrix, NMH, maxiter)
    score_list = []
    score_sum = 1
    for i in range(len(total_path_list)):
        for j in range(len(total_path_list[i])):
            if j != len(total_path_list[i])-1:
                score_sum = score_sum * NMH[total_path_list[i][j], total_path_list[i][j+1]]
        score_list.append(score_sum)
    score_array = np.array(score_list)
    predictscore = np.sum(score_array)

    return predictscore


if __name__ == '__main__':

    # 定义后面要用的超参数
    L = 3  # 路径长度
    maxiter = 10
    # 读取关系数据
    with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
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

        # # 计算相似高斯相似矩阵
        # make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        # circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        # 这里把高斯相似矩阵存一下，方便下次直接读取
        with h5py.File('./Data/GIP_sim_matrix.h5', 'r') as hf:
            circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
            dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # 原文中计算CS是使用circRNA的表达谱
        # 原文中计算DS是使用disease的语义相似性（基于DOSE包）
        CS_matrix = CS(rel_matrix, circ_gipsim_matrix)
        DS_matrix = DS(rel_matrix, dis_gipsim_matrix)

        # 构造异构矩阵H，这里因为没有其他附加数据H矩阵简化了中间行和中间列
        H_matrix = np.vstack((np.hstack((CS_matrix, rel_matrix)), np.hstack((rel_matrix.T, DS_matrix))))

        # 求H矩阵的度矩阵D
        D_matrix = np.zeros((H_matrix.shape[0], H_matrix.shape[1]))
        for i in range(H_matrix.shape[0]):
            row_sum = np.sum(H_matrix[i,:])
            D_matrix[i,i] = row_sum

        # 利用特征值和特征向量求出D矩阵的-1/2次方
        v, Q = np.linalg.eig(D_matrix)
        V = np.diag(v**(-0.5))
        D_half = Q * V * np.linalg.inv(Q)

        NMH = np.dot(D_half, H_matrix, D_half)

        # 开始预测
        prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))

        for i in range(prediction_matrix.shape[0]):
            for j in range(prediction_matrix.shape[1]):
                prediction = predict_sore(i, j+533, L, H_matrix, NMH, maxiter) # 因为关系矩阵在H矩阵上层右侧左边还有一个circRNA的关系矩阵关系矩阵大小为533*533
                prediction_matrix[i,j] = prediction

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

    # 存储tpr，fpr,recall,precision
    with h5py.File('./PlotFigure/BRWSP_AUC.h5') as hf:
        hf['fpr'] = mean_cross_fpr
        hf['tpr'] = mean_cross_tpr
    with h5py.File('./PlotFigure/BRWSP_AUPR.h5') as h:
        h['recall'] = mean_cross_recall
        h['precision'] = mean_cross_precision

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig("roc-KATZCPDA.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()







