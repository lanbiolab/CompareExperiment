import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
# import sklearn.cluster as sc
from MakeSimilarityMatrix import MakeSimilarityMatrix#计算高斯核相似性

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import time

from sklearn.preprocessing import minmax_scale
import pandas as pd

# import xlsxwriter

from models import GraphConv, AE, LP
# from utils import *
# from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.') #原来为default=500
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='Weight between lncRNA space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1, 2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
set_seed(args.seed, args.cuda)
# gdi, ldi, rnafeat, gl, gd = load_data(args.data, args.cuda)


class GNNq(nn.Module):
    def __init__(self):
        super(GNNq, self).__init__()
        self.gnnql = AE(rnafeat.shape[1], 256, args.hidden)
        self.gnnqd = AE(gdi.shape[0], 256, args.hidden)

    def forward(self, xl0, xd0):
        hl, stdl, xl = self.gnnql(gl, xl0)
        hd, stdd, xd = self.gnnqd(gd, xd0)
        return hl, stdl, xl, hd, stdd, xd


class GNNp(nn.Module):
    def __init__(self):
        super(GNNp, self).__init__()
        self.gnnpl = LP(args.hidden, ldi.shape[1])
        self.gnnpd = LP(args.hidden, ldi.shape[0])

    def forward(self, y0):
        yl, zl = self.gnnpl(gl, y0)
        yd, zd = self.gnnpd(gd, y0.t())
        return yl, zl, yd, zd


print("Dataset {}, 5-fold CV".format(args.data))

def GIP(circrna_disease_matrix):
    make_sim_matrix = MakeSimilarityMatrix(circrna_disease_matrix)
    return make_sim_matrix

def find_key(i, cancer_dict):
    name = list(cancer_dict.keys())[list(cancer_dict.values()).index(i)]

    return name


def neighborhood(feat, k):
    # print("This is neighborhood...")
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def normalized(wmat):
    # print("This is normalized...")
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    # print("This is norm_adj...")
    C = neighborhood(feat.T, k=10)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

def load_data(rel_matrix, cuda):
    print("This is load_data...")
    # path = 'Dataset' + str(data)
    # gdi = pd.read_csv(path + '/disease_GaussianSimilarity.csv', header=None, encoding='gb18030').values
    # ldi = pd.read_csv(path + '/association_matrix.csv', header=None, encoding='gb18030').values
    # rnafeat = pd.read_csv(path + '/rna_GaussianSimilarity.csv', header=None, encoding='gb18030').values
    # 计算相似高斯相似矩阵
    make_sim_matrix = GIP(rel_matrix)
    # circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
    rnafeat, gdi = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
    ldi = rel_matrix.copy()


    rnafeat = minmax_scale(rnafeat, axis=0)
    gdit = torch.from_numpy(gdi).float()
    ldit = torch.from_numpy(ldi).float()
    rnafeatorch = torch.from_numpy(rnafeat).float()
    gl = norm_adj(rnafeat)
    gd = norm_adj(gdi.T)
    if cuda:
        gdit = gdit.cuda()
        ldit = ldit.cuda()
        rnafeatorch = rnafeatorch.cuda()
        gl = gl.cuda()
        gd = gd.cuda()

    return gdit, ldit, rnafeatorch, gl, gd

def criterion(output, target, msg, n_nodes, mu, logvar):
    if msg == 'disease':
        cost = F.binary_cross_entropy(output, target)
    else:
        cost = F.mse_loss(output, target)

    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL


def train(gnnq, gnnp, xl0, xd0, y0, epoch, alpha, i):
    losspl1 = []
    losspd1 = []
    lossp1 = []
    lossq1 = []
    beta0 = 1.0
    gamma0 = 1.0
    optp = torch.optim.Adam(gnnp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optq = torch.optim.Adam(gnnq.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(epoch):
        gnnq.train()

        gnnq.train()
        hl, stdl, xl, hd, stdd, xd = gnnq(xl0, xd0)
        lossql = criterion(xl, xl0,
                           "lncrna", gl.shape[0], hl, stdl)
        lossqd = criterion(xd, xd0,
                           "disease", gd.shape[0], hd, stdd)
        lossq = alpha * lossql + (1 - alpha) * lossqd + beta0 * e * F.mse_loss(
            torch.mm(hl, hd.t()), y0) / epoch
        optq.zero_grad()
        lossq1.append(lossq.item())
        lossq.backward()
        optq.step()
        gnnq.eval()
        with torch.no_grad():
            hl, _, _, hd, _, _ = gnnq(xl0, xd0)

        gnnp.train()
        yl, zl, yd, zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl, y0) + gamma0 * e * F.mse_loss(zl, hl) / epoch
        losspd = F.binary_cross_entropy(yd, y0.t()) + gamma0 * e * F.mse_loss(zd, hd) / epoch
        lossp = alpha * losspl + (1 - alpha) * losspd
        losspl1.append(losspl.item())
        losspd1.append(losspd.item())
        lossp1.append(lossp.item())
        optp.zero_grad()
        lossp.backward()
        optp.step()

        with torch.no_grad():
            yl, _, yd, _ = gnnp(y0)
        if e % 20 == 0:
            print('Epoch %d | Lossp: %.4f | Lossq: %.4f' % (e, lossp.item(), lossq.item()))
        # r = pd.DataFrame(lossp1)
        # r.to_csv('./output/lossp1{}.csv'.format(i))
        # r1 = pd.DataFrame(lossq1)
        # r.to_csv('./output/lossq1{}.csv'.format(i))
        # r = pd.DataFrame(losspd1)
        # r.to_csv('./output/losspd1{}.csv'.format(i))
        # r = pd.DataFrame(losspl1)
        # r.to_csv('./output/losspl1{}.csv'.format(i))
    return alpha * yl + (1 - alpha) * yd.t()


if __name__=="__main__":

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circ2Traits/circRNA_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

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
    cancer_dict = {'bladder cancer': 58, 'breast cancer': 46, 'glioma': 89, 'glioblastoma': 88,
                   'glioblastoma multiforme': 59, 'cervical cancer': 23, 'colorectal cancer': 6, 'gastric cancer': 15}
    # # circad
    # cancer_dict = {'bladder cancer':94, 'breast cancer':53, 'triple-negative breast cancer':111, 'gliomas':56, 'glioma':76,
    #                 'cervical cancer':65, 'colorectal cancer':143, 'gastric cancer':28}

    # denovo start
    for i in range(60,90):
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
        roc_circrna_disease_matrix = circrna_disease_matrix.copy()
        if ((False in (new_circrna_disease_matrix[:, i] == 0)) == False):
            continue
        new_circrna_disease_matrix[:, i] = 0
        rel_matrix = new_circrna_disease_matrix

        ############这里应该要改############
        # 计算相似高斯相似矩阵
        # make_sim_matrix =GIP(rel_matrix)
        # circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
        # circnum = circrna_disease_matrix.shape[0]
        # disnum = circrna_disease_matrix.shape[1]
        # SC = circ_gipsim_matrix
        # SD = dis_gipsim_matrix
        # SB = IBNP(circnum, disnum, rel_matrix)
        # # print(SB)
        # SK = KATZ(SC, SD, rel_matrix)
        # S = (SB + SK) / 2

        # rel_matrix = rel_matrix[:2][:2]
        # roc_circrna_disease_matrix = roc_circrna_disease_matrix[:2][:2]

        gdi, ldi, rnafeat, gl, gd = load_data(rel_matrix, args.cuda)
        # print("gdi",gdi)
        # gdi = torch.tensor(gdi)
        # ldi = torch.tensor(ldi)
        # rnafeat = torch.tensor(rnafeat)
        # gl = torch.tensor(gl)
        # gd = torch.tensor(gd)

        print("load_data完成")

        gnnq = GNNq()
        gnnp = GNNp()
        if args.cuda:
            gnnq = gnnq.cuda()
            gnnp = gnnp.cuda()

        rel_matrix_tensor = torch.tensor(np.array(rel_matrix).astype(np.float32))
        train(gnnq, gnnp, rnafeat, gdi.t(), rel_matrix_tensor, args.epochs, args.alpha, i)
        gnnq.eval()
        gnnp.eval()
        yli, _, ydi, _ = gnnp(rel_matrix_tensor)
        resi = args.alpha * yli + (1 - args.alpha) * ydi.t()
        if args.cuda:
            ymat = resi.cpu().detach().numpy()
        else:
            ymat = resi.detach().numpy()

        S = ymat
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
        print("roc_auc", np.trapz(tpr_arr_epoch, fpr_arr_epoch))
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
    with h5py.File('./GMNN2CD_circ2Traits_denovo_result_60_90.h5', 'w') as hf:
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


