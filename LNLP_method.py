# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : CD-LNLP method


import numpy as np


def fast_calculate_new(feature_matrix, neighbor_num):
    """
    :param feature_matrix:
    :param neighbor_num: neighbor_num: must be less or equal than n-1 !!!!(n is the row count of feature matrix
    :return:
    """
    iteration_max = 50
    mu = 6
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    temp = alpha + alpha.T - 2 * X * X.T
    temp[np.where(temp < 0)] = 0
    distance_matrix = np.sqrt(temp)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return W


def calculate_linear_neighbor_simi(feature_matrix, neighbor_rate):
    """
    :param feature_matrix:
    :param neighbor_rate:
    :return:
    """
    neighbor_num = int(neighbor_rate * feature_matrix.shape[0])
    return fast_calculate_new(feature_matrix, neighbor_num)


def normalize_by_divide_rowsum(simi_matrix):
    simi_matrix_copy = np.matrix(simi_matrix, copy=True)
    for i in range(simi_matrix_copy.shape[0]):
        simi_matrix_copy[i, i] = 0
    row_sum_matrix = np.sum(simi_matrix_copy, axis=1)
    result = np.divide(simi_matrix_copy, row_sum_matrix)
    result[np.where(row_sum_matrix == 0)[0], :] = 0
    return result


def complete_linear_neighbor_simi_matrix(train_association_matrix, neighbor_rate):
    b = np.matrix(train_association_matrix)
    final_simi = calculate_linear_neighbor_simi(b, neighbor_rate)
    normalized_final_simi = normalize_by_divide_rowsum(
        final_simi)
    return normalized_final_simi


def linear_neighbor_predict(train_matrix, alpha, neighbor_rate, circRNA_weight):
    rna_number = train_matrix.shape[0]
    disease_number = train_matrix.shape[1]
    w_rna = complete_linear_neighbor_simi_matrix(train_matrix, neighbor_rate)
    w_disease = complete_linear_neighbor_simi_matrix(train_matrix.T, neighbor_rate)
    w_rna_eye = np.eye(rna_number)
    w_disease_eye = np.eye(disease_number)
    temp0 = w_rna_eye - alpha * w_rna

    try:
        temp1 = np.linalg.inv(temp0)
    except Exception:
        temp1 = np.linalg.pinv(temp0)
    temp2 = np.dot(temp1, train_matrix)
    prediction_rna = (1 - alpha) * temp2
    temp3 = w_disease_eye - alpha * w_disease
    try:
        temp4 = np.linalg.inv(temp3)
    except Exception:
        temp4 = np.linalg.pinv(temp3)
    temp5 = np.dot(temp4, train_matrix.T)
    temp6 = (1 - alpha) * temp5
    prediction_disease = temp6.T
    prediction_result = circRNA_weight * prediction_rna + (1 - circRNA_weight) * prediction_disease
    return prediction_result
