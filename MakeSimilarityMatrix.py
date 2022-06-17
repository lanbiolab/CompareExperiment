
'''
@Author: Dong Yi
@Date: 2020/5/10
@Description: 这个是根据circRNA-disease关系矩阵，计算高斯相似性，
构成circRNA-circRNA关系矩阵，disease-disease关系矩阵
'''
import h5py
import numpy as np


def mean_matrix_length(tempmatrix):
    sum = []
    for i in range(tempmatrix.shape[0]):
        sum.append(np.linalg.norm(tempmatrix[i, :]) ** 2)
    mean_length = np.mean(np.array(sum))

    return mean_length


# 求circ-circ相似度，或者求disease-disease两个向量之间的高斯相似度
def compute_similarity(row, col, tempmatrix):
    mean_vertex_length = mean_matrix_length(tempmatrix)
    gama = 1 / mean_vertex_length
    vertex1 = tempmatrix[row, :]
    vertex2 = tempmatrix[col, :]
    similarity = np.exp(-gama * np.linalg.norm(vertex1 - vertex2) ** 2)

    return similarity


def circ_similarity(circ):
    circ_simmatrix = np.zeros((circ.shape[0], circ.shape[0])).astype('float32')
    for row in range(circ_simmatrix.shape[0]):
        for col in range(circ_simmatrix.shape[1]):
            circ_simmatrix[row][col] = compute_similarity(row, col, circ)

    return circ_simmatrix


# 求dis与dis的相似度矩阵的功能函数
def dis_similarity(disx):
    dis_simmatrix = np.zeros((disx.shape[0], disx.shape[0])).astype('float32')
    for row in range(dis_simmatrix.shape[0]):
        for col in range(dis_simmatrix.shape[1]):
            dis_simmatrix[row][col] = compute_similarity(row, col, disx)

    return dis_simmatrix

class MakeSimilarityMatrix(object):
    def __init__(self,circrna_disease_matrix):
        self.circsimmatrix ,self.dissimmatrix = self.makesimmatrix(circrna_disease_matrix)


    def makesimmatrix(self,circrna_disease_matrix):
        circ_sim_matrix = circ_similarity(circrna_disease_matrix)
        dis_sim_matrix = dis_similarity(circrna_disease_matrix.transpose())
        return circ_sim_matrix,dis_sim_matrix

