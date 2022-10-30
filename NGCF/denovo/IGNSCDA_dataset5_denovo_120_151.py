import math
import random
import h5py
import numpy as np
# from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2


import sortscore

import time


def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__=="__main__":
    #####数据集1####
    # num_users = 533
    # num_items = 89
    #####数据集1####
    #####数据集2####
    # num_users = 514
    # num_items = 62
    #####数据集2####
    # #####数据集3####
    # num_users = 312
    # num_items = 40
    #####数据集3####
    #####数据集4####
    # num_users = 923
    # num_items = 104
    # #####数据集4####
    #####数据集5####
    num_users = 1265
    num_items = 151
    #####数据集5####

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

    # circ2Traits
    # cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
    #                'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6, 'gastric cancer': 15}
    # # circad
    cancer_dict = {'bladder cancer':94, 'breast cancer':53, 'triple-negative breast cancer':111, 'gliomas':56, 'glioma':76,
                    'cervical cancer':65, 'colorectal cancer':143, 'gastric cancer':28}

    # denovo start
    # foldi = 1
    for i in range(120,151): #num_items
        print("#i=",i)

        ############这里应该要改############
        FeaturePath = '../Feature/denovo/dataset5_gcn_embedding_128_feature_fold%d.h5' % (i+1)  # 注意更改

        ###########################更改读文件的方式#######################################
        train_exist_users = []
        train_exist_items = []
        trainfile_path = '../../Data/dataset5/denovo/circRNA-disease-fold%d/train.txt' % (i+1)  # 注意更改
        testfile_path = '../../Data/dataset5/denovo/circRNA-disease-fold%d/test.txt' % (i+1)  # 注意更改

        # foldi += 1

        with open(trainfile_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(ii) for ii in l[1:]]
                    uid = int(l[0])
                    train_exist_users.append(uid)
                    train_exist_items.append(items)
                    # print("items", items)
                    # print("uid", uid)

        train_u_nodes = []
        train_v_nodes = []
        # train_ratings = []
        train_index = []

        for ii in range(len(train_exist_users)):
            temp_user = train_exist_users[ii]
            for j in range(len(train_exist_items[ii])):
                temp_item = train_exist_items[ii][j]
                train_u_nodes.append(temp_user)
                train_v_nodes.append(temp_item)
                # train_ratings.append(1)
                train_index.append((temp_user, temp_item))

        # print("###########")

        test_exist_users = []
        test_exist_items = []
        with open(testfile_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(ii) for ii in l[1:]]
                    uid = int(l[0])
                    test_exist_users.append(uid)
                    test_exist_items.append(items)
                    # print("items", items)
                    # print("uid", uid)

        test_u_nodes = []
        test_v_nodes = []
        # test_ratings = []
        test_index = []

        for ii in range(len(test_exist_users)):
            temp_user = test_exist_users[ii]
            for j in range(len(test_exist_items[ii])):
                temp_item = test_exist_items[ii][j]
                test_u_nodes.append(temp_user)
                test_v_nodes.append(temp_item)
                # test_ratings.append(1)
                test_index.append((temp_user, temp_item))

        circrna_disease_matrix = np.zeros((num_users, num_items))
        for ii in range(len(train_exist_users)):
            temp_user = train_exist_users[ii]
            for j in range(len(train_exist_items[ii])):
                temp_item = train_exist_items[ii][j]
                circrna_disease_matrix[temp_user][temp_item] = 1

        for ii in range(len(test_exist_users)):
            temp_user = test_exist_users[ii]
            for j in range(len(test_exist_items[ii])):
                temp_item = test_exist_items[ii][j]
                circrna_disease_matrix[temp_user][temp_item] = 1

        # new_circrna_disease_matrix = circrna_disease_matrix.copy()
        ############################更改读文件的方式######################################

        # 根据每次fold的值不一样，读取特征的h5文件不一样
        # with h5py.File('./flatten_feature_h5_file/flatten_layer_output_in_fold%d.h5'%fold, 'r') as f:
        #     cnn_pair_relation_feature = f['flatten_layer_output'][:]
        with h5py.File(FeaturePath, 'r') as f:
            gcn_circRNA_feature = f['user_feature'][:]
            gcn_disease_feature = f['item_feature'][:]

        # 把一部分已知关系置零
        # test_index = one_list[i:i+split]
        # train_index = list(set(one_list)-set(test_index))
        # # 这里把五次的每一次训练下标和测试下标给记下来
        # with h5py.File('./one_list_file/train_test_fold%d.h5'%fold) as f:
        #     f['train_index'] = train_index
        #     f['test_index'] = test_index


        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        # roc_circrna_disease_matrix = roc_circrna_disease_matrix[:2][:2]
        # rel_matrix = rel_matrix[:2][:2]

        # 获取训练集和测试集
        input_fusion_feature_x = []
        input_fusion_x_label = []
        for (u, ii) in train_index:
            # 正样本
            # 取出对应的CNN和GCN中的特征向量
            # GCN的circRNA部分192维
            gcn_circRNA_array = gcn_circRNA_feature[u, :]
            # GCN的disease部分192维
            gcn_disease_array = gcn_disease_feature[ii, :]
            # CNN的pair关系部分192维
            # cnn_pair_array = cnn_pair_relation_feature[u*rel_matrix.shape[1]+i,:]
            fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
            # fusion_feature = np.multiply(gcn_circRNA_array,gcn_disease_array)
            input_fusion_feature_x.append(fusion_feature.tolist())
            input_fusion_x_label.append(1)
            # 负样本
            for num in range(4):
                j = np.random.randint(rel_matrix.shape[1])
                while (u, j) in train_index:
                    j = np.random.randint(rel_matrix.shape[1])
                gcn_disease_array = gcn_disease_feature[j, :]
                # cnn_pair_array = cnn_pair_relation_feature[u*rel_matrix.shape[1]+j,:]
                fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
                # fusion_feature = np.multiply(gcn_circRNA_array, gcn_disease_array)
                input_fusion_feature_x.append(fusion_feature.tolist())
                input_fusion_x_label.append(0)

        input_fusion_feature_test_x = []
        input_fusion_test_x_label = []
        # 测试集构造，同样每个正样本选择一个负样本作为测试集
        for row in range(rel_matrix.shape[0]):
            for col in range(rel_matrix.shape[1]):
                # GCN circRNA 128维
                gcn_circRNA_array = gcn_circRNA_feature[row, :]
                # GCN的disease部分128维
                gcn_disease_array = gcn_disease_feature[col, :]
                # CNN的pair关系部分192维
                # cnn_pair_array = cnn_pair_relation_feature[row*rel_matrix.shape[1]+col,:]
                fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
                # fusion_feature = np.multiply(gcn_circRNA_array, gcn_disease_array)
                input_fusion_feature_test_x.append(fusion_feature.tolist())
                input_fusion_test_x_label.append(rel_matrix[row, col])

        # 构造神经网络
        model = Sequential()
        model.add(Dense(512, input_shape=(512,), kernel_regularizer=l2(0.0001), activation='relu', name='dense1'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0001), name='dense2'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0001), name='dense3'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', name='prediction'))
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=250, batch_size=100)
        # history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=2,
        #                     batch_size=10)  # 注意更改
        predictions = model.predict(np.array(input_fusion_feature_test_x), batch_size=100)
        # 把这个预测prediction拟合为533*89的形式
        prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))
        predictions_index = 0
        for row in range(prediction_matrix.shape[0]):
            for col in range(prediction_matrix.shape[1]):
                prediction_matrix[row, col] = predictions[predictions_index]
                predictions_index += 1
        # print(prediction_matrix.shape)
        # print(roc_circrna_disease_matrix.shape)

        score_matrix_temp = prediction_matrix.copy()

        S = score_matrix_temp
        ############这里应该要改############

        prediction_matrix = S#得到预测矩阵，这个预测矩阵中的数值为0-1之间的数
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
        print("##i=",i)
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

        ################分割线################
        tpr_arr_epoch = np.array(tpr_list)
        fpr_arr_epoch = np.array(fpr_list)
        recall_arr_epoch = np.array(recall_list)
        precision_arr_epoch = np.array(precision_list)
        accuracy_arr_epoch = np.array(accuracy_list)
        F1_arr_epoch = np.array(F1_list)
        # print("epoch,",epoch)
        print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (
            np.mean(accuracy_arr_epoch), np.mean(recall_arr_epoch), np.mean(precision_arr_epoch),
            np.mean(F1_arr_epoch)))
        print("roc_auc", np.trapz(tpr_arr_epoch, fpr_arr_epoch))
        print("AUPR", np.trapz(precision_arr_epoch, recall_arr_epoch))

        # print("TP=%d, FP=%d, TN=%d, FN=%d" % (TP, FP, TN, FN))
        # print("roc_auc", np.trapz(tpr_arr_epoch, fpr_arr_epoch))
        ##############分割线######################

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
    with h5py.File('./IGNSCDA_dataset5_denovo_result_120_151.h5', 'w') as hf:
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
    print("runtime over, now is :")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()


