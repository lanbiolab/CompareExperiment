'''
@Author: Dong Yi
@Date: 2020.11.23
@Description: 这是对BRWSP的denovo实验
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

def find_max_prop(prop_list):
    maxindex_list = []
    max_of_prop = max(prop_list)
    for i in range(len(prop_list)):
        if max_of_prop == prop_list[i]:
            maxindex_list.append(i)

    return maxindex_list


def find_nextnode_list(current_node, prev_node, path_length, temp_path, H_matrix):
    current_node_neighborlist = []
    # 判断这个路径长度
    if path_length==0 : # 代表这个结点是第一个节点是论文中k=1的情况
        current_neighborhood_list = find_neighborhood(current_node, H_matrix)
        next_node = current_neighborhood_list[-1]
        current_node_neighborlist.append(next_node)
    else: # 代表这个节点不是第一个节点
        # 去寻找当前节点和前一个节点的邻接节点list
        current_neighborhood_list = find_neighborhood(current_node, H_matrix)
        prev_neighborhood_list = find_neighborhood(prev_node, H_matrix)
        prop_list = compute_neighbor_prop(current_neighborhood_list, prev_neighborhood_list, current_node, prev_node, NMH, temp_path)
        # 这个prop_list可能最大值有多个把他们全部找出来
        max_prop_index = find_max_prop(prop_list)
        for i in range(len(max_prop_index)):
            current_node_neighborlist.append(current_neighborhood_list[max_prop_index[i]])

    return current_node_neighborlist


# 注意这个函数是非常重要的函数，就是在这里进行BFS和DFS的结合
def find_path(start, destination, L, H_matrix, NMH, maxiter):
    # 定义搜集起始节点到终点的所有路径
    total_path_list = []
    for i in range(maxiter):
        current_node = start
        prev_node = start
        temp_path = []
        queue = []
        seen_node = set()
        path_length = -1

        queue.append(current_node)
        seen_node.add(current_node)
        while len(queue)>0 and path_length < L:
            # 开始遍历队列弹出第一个节点
            prev_node = current_node
            current_node = queue.pop(0)
            temp_path.append(current_node)
            path_length += 1
            if current_node == destination:
                break;
            # 找到当前这个节点的邻居节点
            next_node_list = find_nextnode_list(current_node, prev_node, path_length, temp_path, H_matrix)
            # 将当前邻居结点随机选择一个（这是为什么要重复那么多次的原因）
            ran_current_node_list = random.sample(next_node_list, 1)
            for w in ran_current_node_list:
                if w not in seen_node:
                    queue.append(w)
                    seen_node.add(w)
                else: # 如果选中的这个节点出现在seen_node,即之前出现过，则重新选择一遍，避免环路出现
                    ran_current_node_list = random.sample(next_node_list, 1)
        if temp_path[-1] != destination:
            continue
        else:
            total_path_list.append(temp_path)
    # 对total_path_list去重
    final_path_list = []
    for item in total_path_list:
        if item not in final_path_list:
            final_path_list.append(item)

    return final_path_list


# 从这里开始预测从起点i 到 终点 j的预测值
def predict_sore(start, destination, L, H_matrix, NMH, maxiter):

    total_path_list = find_path(start, destination, L, H_matrix, NMH, maxiter)
    score_list = []
    score_sum = 1
    for i in range(len(total_path_list)):
        for j in range(len(total_path_list[i])):
            if j != len(total_path_list[i])-1:
                score_sum = score_sum * NMH[total_path_list[i][j], total_path_list[i][j+1]]
        # 要把这一条路径score_sum求次幂,其中的1是alpha=1
        score_sum = score_sum**(1*(len(total_path_list[i])-1))
        score_list.append(score_sum)
    score_array = np.array(score_list)
    predictscore = np.sum(score_array)

    return predictscore

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':

    # 定义后面要用的超参数
    L = 3  # 路径长度
    maxiter = 10
    # 读取关系数据
    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]
    #
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
    for i in range(135,140):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        # 计算相似高斯相似矩阵
        make_sim_matrix = MakeSimilarityMatrix(rel_matrix)
        circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix

        # # 这里把高斯相似矩阵存一下，方便下次直接读取
        # with h5py.File('./Data/GIP_sim_matrix.h5', 'r') as hf:
        #     circ_gipsim_matrix = hf['circ_gipsim_matrix'][:]
        #     dis_gipsim_matrix = hf['dis_gipsim_matrix'][:]

        # 原文中计算CS是使用circRNA的表达谱
        # 原文中计算DS是使用disease的语义相似性（基于DOSE包）
        CS_matrix = CS(rel_matrix, circ_gipsim_matrix)
        DS_matrix = DS(rel_matrix, dis_gipsim_matrix)

        # 构造异构矩阵H，这里因为没有其他附加数据H矩阵简化了中间行和中间列
        H_matrix = np.vstack((np.hstack((CS_matrix, rel_matrix)), np.hstack((rel_matrix.T, DS_matrix))))

        # 求H矩阵的度矩阵D
        D_matrix = np.zeros((H_matrix.shape[0], H_matrix.shape[1]))
        for m in range(H_matrix.shape[0]):
            row_sum = np.sum(H_matrix[m,:])
            D_matrix[m,m] = row_sum

        # 利用特征值和特征向量求出D矩阵的-1/2次方
        v, Q = np.linalg.eig(D_matrix)
        V = np.diag(v**(-0.5))
        D_half = Q * V * np.linalg.inv(Q)

        NMH = np.dot(D_half, H_matrix, D_half)

        # 开始预测
        prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))

        for m in range(prediction_matrix.shape[0]):
            for n in range(prediction_matrix.shape[1]):
                prediction = predict_sore(m, n+533, L, H_matrix, NMH, maxiter) # 因为关系矩阵在H矩阵上层右侧左边还有一个circRNA的关系矩阵关系矩阵大小为533*533
                prediction_matrix[m,n] = prediction

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
    with h5py.File('./BRWSP_denovo_result_135_140.h5', 'w') as hf:
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
    # with h5py.File('./PlotFigure/BRWSP_circ2Traits_denovo_AUC.h5') as hf:
    #     hf['fpr'] = mean_denovo_fpr
    #     hf['tpr'] = mean_denovo_tpr
    # with h5py.File('./PlotFigure/BRWSP_circ2Traits_denovo_AUPR.h5') as h:
    #     h['recall'] = mean_denovo_recall
    #     h['precision'] = mean_denovo_precision
    #
    # plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean denovo ROC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.savefig("./FinalResultPng/roc-BRWSP_small_circ2Traits_denovo.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()


