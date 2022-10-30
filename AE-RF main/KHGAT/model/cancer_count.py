# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import os
from matplotlib import pyplot
from numpy import interp
# import matplotlib.pyplot as plt
import sklearn
# import xlsxwriter
# import xlrd
from sklearn import svm  # , grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import h5py
import pdb
import random
from random import randint
import scipy.io

# from keras.layers import merge

from tensorflow.keras.layers import Input, Dense
# from keras.engine.training import Model
from tensorflow.keras.models import Sequential, model_from_config, Model
# from keras.layers.core import Dropout, Activation, Flatten  # , Merge
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
    # , generic_utils
# from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
# from keras.layers import normalization
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.recurrent import LSTM
# from keras.layers.embeddings import Embedding
# from keras import regularizers
# from keras.constraints import maxnorm
import sortscore
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
# from keras.layers import containers, normalization


def prepare_data(fold, seperate=False):
    print("loading data")

    circRNA_fea = np.loadtxt("integrated_circRNA_similarity5.txt", dtype=float, delimiter="\t")
    disease_fea = np.loadtxt("integrated_disease_similarity5.txt", dtype=float, delimiter="\t")
    pca = PCA(n_components=80)  # n_components 选择降维数量
    pca = pca.fit(circRNA_fea)  # 开始拟合建模
    circRNA_fea = pca.transform(circRNA_fea)  # 获得降维后数据
    pca_ = PCA(n_components=80)  # n_components 选择降维数量
    pca_ = pca_.fit(disease_fea)  # 开始拟合建模
    disease_fea = pca_.transform(disease_fea)  # 获得降维后数据
    # interaction = np.loadtxt("associations.txt", dtype=float, delimiter=",")
    df = pd.read_excel('associations5.xls', 'sheet1', header=None)
    interaction = np.array(df)
    a = [0]*interaction.shape[0]
    interaction[:,fold] = np.array(a).T

    link_number = 0
    # nonlink_number=0
    train = []
    test = []
    testfnl = []
    label1 = []
    label2 = []
    label22 = []
    ttfnl = []
    label = []
    # link_position = []
    # nonLinksPosition = []

    for i in range(0,interaction.shape[0]):  # shape[0] returns m if interaction is m*n, ie, returns no. of rows of matrix
        for j in range(0, interaction.shape[1]):
            # print(interaction.shape[0])
            if interaction[i, j] == 1:  # for associated
                label1.append(interaction[i, j])  # label1= labels for association(1)
                link_number = link_number + 1  # no. of associated samples
                # link_position.append([i, j])
                circRNA_fea_tmp = list(circRNA_fea[i])
                disease_fea_tmp = list(disease_fea[j])
                tmp_fea = (circRNA_fea_tmp, disease_fea_tmp)  # concatnated feature vector for an association
                train.append(tmp_fea)  # train contains feature vectors of all associated samples
            elif interaction[i, j]==0:
                label2.append(interaction[i, j])  # label2= labels for no association(0)
                # nonlink_number = nonlink_number + 1
                # nonLinksPosition.append([i, j])
                # print(i)
                circRNA_fea_tmp1 = list(circRNA_fea[i])
                disease_fea_tmp1 = list(disease_fea[j])
                test_fea = (circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                testfnl.append(test_fea)  # testfnl contains feature vectors of all non associated samples
            label.append(interaction[i, j])  # label2= labels for no association(0)
            # nonlink_number = nonlink_number + 1
            # nonLinksPosition.append([i, j])
            # print(i)
            circRNA_fea_ = list(circRNA_fea[i])
            disease_fea_ = list(disease_fea[j])
            test_fea_ = (circRNA_fea_, disease_fea_)  # concatenated feature vector for not having association
            test.append(test_fea_)  # testfnl contains feature vectors of all non associated samples


    print("link_number", link_number)

    m = np.arange(len(label2))
    np.random.shuffle(m)

    for x in m:
        ttfnl.append(testfnl[x])
        label22.append(label2[x])
    # print('************')
    # print(ttfnl)
    # print('************')
    # print(label22)
    for x in range(0, link_number):  # for equalizing positive and negative samples
    # for x in range(0, len(ttfnl)):
        tfnl = ttfnl[x]  # tfnl= feature vector pair for no association
        lab = label22[x]  # lab= label of the above mentioned feature vector pair(0)
        # print(tfnl)
        # print('***')
        train.append(tfnl)  # append the non associated feature vector pairs to train till x<=no. of associated pairs
        label1.append(lab)  # append the labels of non associated pairs(0) to label1

    # print(train)
    # print(label1)
    return np.array(train), label1, label, np.array(test), interaction, link_number


def calculate_performace(test_num, pred_y, labels):  # pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)

    return acc, precision, sensitivity, specificity, MCC, f1_score


def transfer_array_format(data):  # data=X  , X= all the circRNA features, disease features
    formated_matrix1 = []
    formated_matrix2 = []
    # pdb.set_trace()
    # pdb.set_trace()
    for val in data:
        # formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])  # contains circRNA features ?
        formated_matrix2.append(val[1])  # contains disease features ?
        # formated_matrix1[0] = np.array([val[0]])
        # formated_matrix2.append(np.array([val[1]]))
        # formated_matrix2[0] = val[1]

    return np.array(formated_matrix1), np.array(formated_matrix2)


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def DNN_auto(x_train):
    encoding_dim = 128  # 128 original
    input_img = Input(shape=(160,))

    encoded = Dense(1050, activation='relu')(input_img)  # 450 - output (input layer)
    # encoded = Dense(250, activation='relu')(encoded)     # 200 - output (hidden layer1)
    encoded = Dense(250, activation='relu')(encoded)  # 100 - output (hidden layer2)
    encoder_output = Dense(encoding_dim)(encoded)  # 128 - output (encoding layer)
    # print()
    # decoder layers
    decoded = Dense(1050, activation='relu')(encoder_output)
    # decoded = Dense(250, activation='relu')(decoded)
    decoded = Dense(250, activation='relu')(decoded)
    decoded = Dense(160, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder = Model(inputs=input_img, outputs=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train, epochs=20, batch_size=20,shuffle=True)  # second x_train is given instead of train labels in DNN, ie here, i/p=o/p

    # batch_size=100 original
    encoded_imgs = encoder.predict(x_train)
    return encoded_imgs


def DeepCDA():


    # encoder,X_data2 = DNN_auto(X_data2)

    # num_cross_val = 10
    all_performance_rf = []
    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []
    all_performance_DNN = []
    # print(y)
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []
    # df = pd.read_excel('associations2.xls', 'sheet1', header=None)
    # interaction = np.array(df)
    fold = 28
    if True:
        X, labels, label, T, interaction, link_number = prepare_data(fold,
            seperate=True)  # X= array of concatinated features,labels=corresponding labels
        # import pdb            #python debugger
        df = pd.read_excel('associations5.xls', 'sheet1', header=None)
        interaction = np.array(df)
        X_data1, X_data2 = transfer_array_format(
            X)  # X-data1 = circRNA features(2500*495),  X_data2 = disease features (2500*383)
        t1, t2 = transfer_array_format(
            T)  # X-data1 = circRNA features(2500*495),  X_data2 = disease features (2500*383)

        print("************")
        print(X_data1.shape, X_data2.shape)  # (36352,512), (36352,71)
        print("******************")

        X_data1 = np.concatenate((X_data1, X_data2), axis=1)  # axis=1 , rowwoise concatenation
        T_data = np.concatenate((t1, t2), axis=1)
        print("************")
        print(X_data1.shape)  # (36352,583)
        print("******************")

        y, encoder = preprocess_labels(labels)  # labels labels_new
        num = np.arange(len(X_data1))  # num gets an array like num = [0,1,2...len(y)], len(y) = 512*71 = 36352
        np.random.shuffle(num)
        X_data1 = X_data1[num]
        X_data2 = X_data2[num]
        y = y[num]

        t = 0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        X_data1 = DNN_auto(X_data1)  # Now X_data1 contains Auto encoded output
        T_data = DNN_auto(T_data)
        # train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        train1 = np.array(X_data1)
        test1 = np.array([x for i, x in enumerate(T_data)])
        # train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        # test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = y
        test_label = np.array(label)
        # print("$$$$$$$$$$$$",test1)
        # print(test2)

        # real_labels = []
        # for val in test_label:
        #     if val[0] == 1:  # tuples in array, val[0]- first element of tuple
        #         real_labels.append(0)
        #     else:
        #         real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        class_index = 0

        ## DNN
        class_index = class_index + 1
        #        prefilter_train = np.concatenate((train1, train2), axis = 1)
        #        prefilter_test = np.concatenate((test1, test2), axis = 1)

        prefilter_train = train1
        prefilter_test = test1

        clf = RandomForestClassifier(n_estimators=100,random_state=40)

        clf.fit(prefilter_train, train_label_new)  # ***Training

        ae_y_pred_prob = clf.predict_proba(test1)[:, 1]  # **testing
        # proba = []
        # for i in range(len(ae_y_pred_prob)//2):
        #     proba.append((ae_y_pred_prob[i]+ae_y_pred_prob[i+len(ae_y_pred_prob)//2])/2)

        # print(ae_y_pred_prob)
        dict={}
        count=0
        for k in range(interaction.shape[0]):
            for l in range(interaction.shape[1]):
                dict[count]=(k,l)
                count+=1
        matrix = interaction.copy()
        a = [0]*interaction.shape[0]
        matrix[:,fold]=np.array(a).T
        matrix=matrix+interaction
        interaction_1 = np.zeros(interaction.shape)
        for k in range(interaction.shape[0]):
            for l in range(interaction.shape[1]):
                interaction_1[k][l]=ae_y_pred_prob[k*l+l]

        # count = 0
        # interaction_2 = interaction.copy()
        # for k in range(interaction.shape[0]):
        #     for l in range(interaction.shape[1]):
        #         if interaction[k][l]==1 and count%5==fold:
        #             interaction[k][l] = proba[count]
        #             interaction_2[k][l]=0
        #             count+=1
        # roc_circrna_disease_matrix = interaction_2+interaction
        # prediction_matrix = interaction.copy()
        # count = 0
        # for k in range(interaction.shape[0]):
        #     for l in range(interaction.shape[1]):
        #         if interaction[k][l]==1:
        #             # interaction[k][l] = proba[count]
        #             prediction_matrix[k][l] = proba[count]
        #             count += 1
        zero_matrix = np.zeros(interaction.shape)
        score_matrix_temp = interaction_1.copy()
        score_matrix = score_matrix_temp + zero_matrix  # 标签矩阵等于预测矩阵加抹除关系的矩阵
        minvalue = np.min(score_matrix)  # 每列中的最小值
        score_matrix[np.where(matrix == 2)] = minvalue - 20  # ？
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix, matrix)
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:10,fold]==1))
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:20, fold]==1))
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:50, fold]==1))
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:100, fold]==1))
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:150, fold]==1))
        print(np.count_nonzero(sorted_circrna_disease_matrix[0:200, fold]==1))


    #     tpr_list = []
    #     fpr_list = []
    #     recall_list = []
    #     precision_list = []
    #     accuracy_list = []
    #     F1_list = []
    #     for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
    #         P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
    #         N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
    #         TP = np.sum(P_matrix == 1)
    #         FP = np.sum(P_matrix == 0)
    #         TN = np.sum(N_matrix == 0)
    #         FN = np.sum(N_matrix == 1)
    #         tpr = TP / (TP + FN)
    #         fpr = FP / (FP + TN)
    #         tpr_list.append(tpr)
    #         fpr_list.append(fpr)
    #         recall = TP / (TP + FN)
    #         precision = TP / (TP + FP)
    #         recall_list.append(recall)
    #         precision_list.append(precision)
    #         accuracy = (TN + TP) / (TN + TP + FN + FP)
    #         # print(TP, FP, FN)
    #         F1 = (2 * TP) / (2 * TP + FP + FN)
    #         # print(F1)
    #         accuracy_list.append(accuracy)
    #         F1_list.append(F1)
    #     all_tpr.append(tpr_list)
    #     all_fpr.append(fpr_list)
    #     all_recall.append(recall_list)
    #     all_precision.append(precision_list)
    #     all_accuracy.append(accuracy_list)
    #     all_F1.append(F1_list)
    #
    # tpr_arr = np.array(all_tpr)
    # fpr_arr = np.array(all_fpr)
    # recall_arr = np.array(all_recall)
    # precision_arr = np.array(all_precision)
    # accuracy_arr = np.array(all_accuracy)
    # F1_arr = np.array(all_F1)
    #
    # mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    # mean_cross_fpr = np.mean(fpr_arr, axis=0)
    # mean_cross_recall = np.mean(recall_arr, axis=0)
    # mean_cross_precision = np.mean(precision_arr, axis=0)
    # mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
    # # 计算此次五折的平均评价指标数值
    # mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    # mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    # mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    # mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    # print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))
    #
    # # print("%.4f"%2*mean_recall*mean_precision/(mean_recall+mean_precision))
    # roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    # AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    #
    # # 存储tpr，fpr,recall,precision
    # # with h5py.File(r'E:\pythonProject\KHGAT\data\associations4_auc.h5') as hf:
    # #     hf['fpr'] = mean_cross_fpr
    # #     hf['tpr'] = mean_cross_tpr
    # # with h5py.File(r'E:\pythonProject\KHGAT\data\associations4_aupr.h5') as h:
    # #     h['recall'] = mean_cross_recall
    # #     h['precision'] = mean_cross_precision
    #
    # print(roc_auc, AUPR)
    # plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # # plt.savefig("DRGCNCDA.png")
    # print("runtime over, now is :")
    # # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # plt.show()
    #     proba = transfer_label_from_prob(ae_y_pred_prob)
    #     # print(proba)
    #
    #
    #     acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(label), proba,
    #                                                                                    label)
    #
    #     fpr, tpr, auc_thresholds = roc_curve(label, ae_y_pred_prob)
    #     auc_score = auc(fpr, tpr)
    #
    #     ## AUPR score add
    #     precision1, recall, pr_threshods = precision_recall_curve(label, ae_y_pred_prob)
    #     aupr_score = auc(recall, precision1)
    #     print("AUTO-RF:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
    #     all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])
    #     t = t + 1  # AUC fold number
    #
    #     pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
    #     mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
    #     mean_tpr[0] = 0.0
    #
    #     pyplot.xlabel('False positive rate, (1-Specificity)')
    #     pyplot.ylabel('True positive rate,(Sensitivity)')
    #     pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
    #     pyplot.legend()
    #
    # mean_tpr /= num_cross_val
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    #
    # pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)
    # pyplot.legend()
    #
    # pyplot.show()
    #
    # print('*******AUTO-RF*****')
    # print('mean performance of rf using raw feature')
    # print(np.mean(np.array(all_performance_DNN), axis=0))
    # Mean_Result = []
    # Mean_Result = np.mean(np.array(all_performance_DNN), axis=0)
    # print('---' * 20)
    # print('Mean-Accuracy=', Mean_Result[0], '\n Mean-precision=', Mean_Result[1])
    # print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=', Mean_Result[3])
    # print('Mean-MCC=', Mean_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5])
    # print('Mean-Aupr-score=', Mean_Result[6], '\n' 'Mean_F1=', Mean_Result[7])
    # print('---' * 20)


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


if __name__ == "__main__":
    DeepCDA()