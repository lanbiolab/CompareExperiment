'''
@Author:Dong Yi
@Date:2020.7.8
@Description:将CNN提取的特征与GCN提取的特征做一个融合
对融合后的特征进行分类，这里还包含了GCN提取的 特征单独尝试分类
分类器尝试：svm xgboost knn 神经网络
'''
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


#======================================================================正文======================================================#

#============================================接着分类器调整为神经网络做预测==========================================================#


#####数据集1####
num_users = 533
num_items = 89
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
# num_users = 1265
# num_items = 151
#####数据集5####


# 创建五折交叉运算后要记录的数据结构
all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []
all_F1 = []
foldi=1

for ii in range(5):
    print("runtime over, now is :")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    FeaturePath = './Feature/5fold/dataset1_gcn_embedding_128_feature_fold%d.h5' % foldi #注意更改

    ###########################更改读文件的方式#######################################
    train_exist_users = []
    train_exist_items = []
    trainfile_path = '../Data/dataset1/fold_file/circRNA-disease-fold%d/train.txt' % foldi #注意更改
    testfile_path = '../Data/dataset1/fold_file/circRNA-disease-fold%d/test.txt' % foldi #注意更改

    foldi += 1

    with open(trainfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                train_exist_users.append(uid)
                train_exist_items.append(items)
                # print("items", items)
                # print("uid", uid)

    train_u_nodes = []
    train_v_nodes = []
    # train_ratings = []
    train_index = []

    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
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
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                test_exist_users.append(uid)
                test_exist_items.append(items)
                # print("items", items)
                # print("uid", uid)

    test_u_nodes = []
    test_v_nodes = []
    # test_ratings = []
    test_index = []

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
            test_u_nodes.append(temp_user)
            test_v_nodes.append(temp_item)
            # test_ratings.append(1)
            test_index.append((temp_user, temp_item))

    circrna_disease_matrix = np.zeros((num_users, num_items))
    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            circrna_disease_matrix[temp_user][temp_item] = 1

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
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
    roc_circrna_disease_matrix = new_circrna_disease_matrix+circrna_disease_matrix
    rel_matrix = new_circrna_disease_matrix


    # 获取训练集和测试集
    input_fusion_feature_x = []
    input_fusion_x_label=[]
    for (u,i) in train_index:
        # 正样本
        # 取出对应的CNN和GCN中的特征向量
        # GCN的circRNA部分192维
        gcn_circRNA_array = gcn_circRNA_feature[u,:]
        # GCN的disease部分192维
        gcn_disease_array = gcn_disease_feature[i,:]
        # CNN的pair关系部分192维
        # cnn_pair_array = cnn_pair_relation_feature[u*rel_matrix.shape[1]+i,:]
        fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
        # fusion_feature = np.multiply(gcn_circRNA_array,gcn_disease_array)
        input_fusion_feature_x.append(fusion_feature.tolist())
        input_fusion_x_label.append(1)
        # 负样本
        for num in range(20):
            j = np.random.randint(rel_matrix.shape[1])
            while (u,j) in train_index:
                j = np.random.randint(rel_matrix.shape[1])
            gcn_disease_array = gcn_disease_feature[j,:]
            # cnn_pair_array = cnn_pair_relation_feature[u*rel_matrix.shape[1]+j,:]
            fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
            # fusion_feature = np.multiply(gcn_circRNA_array, gcn_disease_array)
            input_fusion_feature_x.append(fusion_feature.tolist())
            input_fusion_x_label.append(0)

    input_fusion_feature_test_x=[]
    input_fusion_test_x_label=[]
    # 测试集构造，同样每个正样本选择一个负样本作为测试集
    for row in range(rel_matrix.shape[0]):
        for col in range(rel_matrix.shape[1]):
            # GCN circRNA 128维
            gcn_circRNA_array = gcn_circRNA_feature[row, :]
            # GCN的disease部分128维
            gcn_disease_array = gcn_disease_feature[col,:]
            # CNN的pair关系部分192维
            # cnn_pair_array = cnn_pair_relation_feature[row*rel_matrix.shape[1]+col,:]
            fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
            # fusion_feature = np.multiply(gcn_circRNA_array, gcn_disease_array)
            input_fusion_feature_test_x.append(fusion_feature.tolist())
            input_fusion_test_x_label.append(rel_matrix[row,col])

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
    # history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=2, batch_size=10) #注意更改
    predictions = model.predict(np.array(input_fusion_feature_test_x), batch_size=100)
    # 把这个预测prediction拟合为533*89的形式
    prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))
    predictions_index = 0
    for row in range(prediction_matrix.shape[0]):
        for col in range(prediction_matrix.shape[1]):
            prediction_matrix[row, col] = predictions[predictions_index]
            predictions_index += 1
    aa = prediction_matrix.shape
    bb = roc_circrna_disease_matrix.shape
    zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
    # print(prediction_matrix.shape)
    # print(roc_circrna_disease_matrix.shape)

    score_matrix_temp = prediction_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20
    sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix, roc_circrna_disease_matrix)

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
print("均值")
print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f"%(mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

print("AUC:%.4f,AUPR:%.4f"%(roc_auc, AUPR))

# disease-circRNA

# GMNN2CD_circRNA_cancer_5fold_AUC
# GMNN2CD_circRNA_cancer_5fold_AUPR

# circRNADisease

# with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
#     circrna_disease_matrix = hf['infor'][:]

# circad
# with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
#     circrna_disease_matrix = hf['infor'][:]
# 存储tpr，fpr,recall,precision
# with h5py.File('./PlotFigure/IGNSCDA_dataset1_5fold_AUC.h5','w') as hf:
#     hf['fpr'] = mean_cross_fpr
#     hf['tpr'] = mean_cross_tpr
# with h5py.File('./PlotFigure/IGNSCDA_dataset1_5fold_AUPR.h5','w') as h:
#     h['recall'] = mean_cross_recall
#     h['precision'] = mean_cross_precision

plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("./FinalResultPng/roc-IGNSCDA_dataset1_5fold.png")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()



