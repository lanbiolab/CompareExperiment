import numpy as np
# import pandas as pd


def sort_matrix(score_matrix,interact_matrix):

    sort_index = np.argsort(-score_matrix,axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:,i] = score_matrix[:,i][sort_index[:,i]]
        # print("score_sorted[:,i]",score_sorted[:,i])
        y_sorted[:,i] = interact_matrix[:,i][sort_index[:,i]]
        # print("y_sorted[:,i]", y_sorted[:, i])
    # print("y_sorted",y_sorted)
    # print("score_sorted", score_sorted)
    # print("y_sorted", y_sorted.shape)
    # print("score_sorted", score_sorted.shape)
    return y_sorted,score_sorted

