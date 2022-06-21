import math
import random
import h5py as h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_core.python.keras import regularizers
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sortscore
from time import time

def DMFCDA(init):

    disease_input = tf.keras.Input(shape=(533,), name='disease_input')
    circRNA_input = tf.keras.Input(shape=(89,), name="circRNA_input")

    left = layers.Dense(533, name="left_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(disease_input)
    left = layers.Dropout(0.005)(left)
    left = layers.Dense(266, name="left_dense_2", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(left)
    left = layers.Dropout(0.005)(left)
    left = layers.Dense(133, name="left_dense_3", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(left)

    right = layers.Dense(89, name="right_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(circRNA_input)
    right = layers.Dropout(0.005)(right)
    right = layers.Dense(44, name="right_dense_2", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(right)
    right = layers.Dropout(0.005)(right)
    right = layers.Dense(22, name="right_dense_3", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(right)

    x = layers.concatenate([left, right], axis=1)

    final_vector = layers.Dense(155, name="final_dense_1", activation="relu", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(x)
    predict = layers.Dense(1, name="prediction_layer", activation="sigmoid", kernel_initializer=init, bias_initializer=init, bias_regularizer=regularizers.l2(0.01))(final_vector)

    model = tf.keras.Model(inputs=[disease_input, circRNA_input], outputs=predict)

    return model


def get_train_set(rel_matrix):
    circRNA_input, disease_input, label = [], [], []
    one_tuple = np.where(rel_matrix == 1)
    one_tuple_list = list(zip(one_tuple[0], one_tuple[1]))
    for (c,d) in one_tuple_list:
        # positive samples
        circRNA_input.append(rel_matrix[c,:])
        disease_input.append(rel_matrix[:,d])
        label.append(1)
        # negative samples
        j = np.random.randint(rel_matrix.shape[1])
        while (c,j) in one_tuple_list:
            j = np.random.randint(rel_matrix.shape[1])
        circRNA_input.append(rel_matrix[c,:])
        disease_input.append(rel_matrix[:,j])
        label.append(0)

    return circRNA_input, disease_input, label

def get_test_set(rel_matrix):
    circRNA_test_input, disease_test_input, label = [], [], []
    for i in range(rel_matrix.shape[0]):
        for j in range(rel_matrix.shape[1]):
            circRNA_test_input.append(rel_matrix[i,:])
            disease_test_input.append(rel_matrix[:,j])
            label.append(rel_matrix[i,j])
    return circRNA_test_input, disease_test_input, label

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name

if __name__ == '__main__':
    num_negatives = 1
    epoches = 100
    batchsize = 100

    # 读取关系数据
    with h5py.File('../../Data/disease-circRNA.h5','r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_miRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('../../Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('../../Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    circrna_num = circrna_disease_matrix.shape[0]
    disease_num = circrna_disease_matrix.shape[1]

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    # 需要特别考虑的六种疾病，记录在字典中
    cancer_dict = {'glioma': 7, 'bladder cancer':9, 'breast cancer': 10,'cervical cancer': 53,'cervical carcinoma': 64,'colorectal cancer':11,'gastric cancer':19}

    # cancer_dict = {'glioma': 23, 'bladder cancer': 2, 'breast cancer': 4, 'cervical cancer': 6,
    #                 'colorectal cancer': 12, 'gastric cancer': 20}

    # cancer_dict = {'glioma': 20, 'bladder cancer': 19, 'breast cancer': 6, 'cervical cancer': 16,
    #                'colorectal cancer': 1, 'gastric cancer': 0}

    # circ2Traits
    # cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
    #                'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6, 'gastric cancer': 15}

    # # circad
    # cancer_dict = {'bladder cancer': 94, 'breast cancer': 53, 'triple-negative breast cancer': 111, 'gliomas': 56,
    #                'glioma': 76,
    #                'cervical cancer': 65, 'colorectal cancer': 143, 'gastric cancer': 28}

    # denovo start
    for i in range(45,50):
        init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        model = DMFCDA(init)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['accuracy'])
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        for epoche in range(epoches):
            t1 = time()
            circRNA_input, disease_input, label = get_train_set(rel_matrix)

            model.fit([np.array(disease_input), np.array(circRNA_input)], np.array(label), epochs=1,
                      batch_size=batchsize, verbose=1, shuffle=True)

            t2 = time()
        circ_test_input, dis_test_input, _ = get_test_set(rel_matrix)
        predictions = model.predict([np.array(dis_test_input), np.array(circ_test_input)], batch_size=100)

        prediction_matrix = np.zeros(rel_matrix.shape)
        for num in range(len(predictions)):
            row_num = num // disease_num
            col_num = num % disease_num
            prediction_matrix[row_num, col_num] = predictions[num][0]
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
    with h5py.File('./results/DMFCDA_denovo_result_45_50.h5', 'w') as hf:
        hf['tpr_arr'] = tpr_arr
        hf['fpr_arr'] = fpr_arr
        hf['recall_arr'] = recall_arr
        hf['precision_arr'] = precision_arr
        hf['accuracy_arr'] = accuracy_arr
        hf['F1_arr'] = F1_arr
