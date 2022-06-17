import numpy as np
import pandas as pd


def sort_matrix(score_matrix,interact_matrix):

    sort_index = np.argsort(-score_matrix,axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:,i] = score_matrix[:,i][sort_index[:,i]]
        y_sorted[:,i] = interact_matrix[:,i][sort_index[:,i]]
    return y_sorted,score_sorted,sort_index

