'''
@File: plot_fig_dataset1.py
@Author: Dong Yi
@Date: 2021/4/7 11:24
@Description:
2021/4/8 这个是对dataset5中所有图的整理
包括 AUC 和 AUPR图的整理
10fold
这个数据中还差
BRWSP 、 GBDTCDA 、 DWNN-RLS、 RWRKNN 这四个模型的数据
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('../PlotFigure/KATZHCDA_circad_10fold_AUC.h5') as f:
    KATZHCDA_tpr = f['tpr'][:]
    KATZHCDA_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/KATZCPDA_circad_10fold_AUC.h5') as f:
    KATZCPDA_tpr = f['tpr'][:]
    KATZCPDA_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/IBNPKATZ_circad_10fold_AUC.h5') as f:
    IBNPKATZ_tpr = f['tpr'][:]
    IBNPKATZ_fpr = f['fpr'][:]
# with h5py.File('../PlotFigure/BRWSP_circ2Traits_AUC.h5') as f:
#     BRWSP_tpr = f['tpr'][:]
#     BRWSP_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/CD-LNLP_circad_10fold_AUC.h5') as f:
    CD_LNLP_tpr = f['tpr'][:]
    CD_LNLP_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/LLCDC_circad_10fold_AUC.h5') as f:
    LLCDC_tpr = f['tpr'][:]
    LLCDC_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/iCircDA_MF_circad_10fold_AUC.h5') as f:
    ICIRCDA_MF_tpr = f['tpr'][:]
    ICIRCDA_MF_fpr = f['fpr'][:]
# with h5py.File('../PlotFigure/RWR-KNN_circad_10fold_AUC.h5') as f:
#     RWR_KNN_tpr = f['tpr'][:]
#     RWR_KNN_fpr = f['fpr'][:]
# with h5py.File('../PlotFigure/GBDTCDA_small_circRNADisease_AUC.h5') as f:
#     GBDTCDA_tpr = f['tpr'][:]
#     GBDTCDA_fpr = f['fpr'][:]
# with h5py.File('../PlotFigure/DWNN-RLS_circ2Traits_AUC.h5') as f:
#     DWNN_RLS_tpr = f['tpr'][:]
#     DWNN_RLS_fpr = f['fpr'][:]
with h5py.File('../PlotFigure/RWR_circad_10fold_AUC.h5') as f:
    RWR_tpr = f['tpr'][:]
    RWR_fpr = f['fpr'][:]


with h5py.File('../PlotFigure/KATZHCDA_circad_10fold_AUPR.h5') as hf:
    KATZHCDA_recall = hf['recall'][:]
    KATZHCDA_precision = hf['precision'][:]
with h5py.File('../PlotFigure/KATZCPDA_circad_10fold_AUPR.h5') as hf:
    KATZCPDA_recall = hf['recall'][:]
    KATZCPDA_precision = hf['precision'][:]
with h5py.File('../PlotFigure/IBNPKATZ_circad_10fold_AUPR.h5') as hf:
    IBNPKATZ_recall = hf['recall'][:]
    IBNPKATZ_precision = hf['precision'][:]
# with h5py.File('../PlotFigure/BRWSP_circ2Traits_AUPR.h5') as hf:
#     BRWSP_recall = hf['recall'][:]
#     BRWSP_precision = hf['precision'][:]
with h5py.File('../PlotFigure/CD-LNLP_circad_10fold_AUPR.h5') as hf:
    CD_LNLP_recall = hf['recall'][:]
    CD_LNLP_precision = hf['precision'][:]
with h5py.File('../PlotFigure/LLCDC_circad_10fold_AUPR.h5') as hf:
    LLCDC_recall = hf['recall'][:]
    LLCDC_precision = hf['precision'][:]
with h5py.File('../PlotFigure/iCircDA_MF_circad_10fold_AUPR.h5') as hf:
    ICIRCDA_MF_recall = hf['recall'][:]
    ICIRCDA_MF_precision = hf['precision'][:]
# with h5py.File('../PlotFigure/RWR-KNN_circad_10fold_AUPR.h5') as hf:
#     RWR_KNN_recall = hf['recall'][:]
#     RWR_KNN_precision = hf['precision'][:]
# with h5py.File('../PlotFigure/GBDTCDA_circ2Traits_AUPR.h5') as hf:
#     GBDTCDA_recall = hf['recall'][:]
#     GBDTCDA_precision = hf['precision'][:]
# with h5py.File('../PlotFigure/DWNN-RLS_circ2Traits_AUPR.h5') as hf:
#     DWNN_RLS_recall = hf['recall'][:]
#     DWNN_RLS_precision = hf['precision'][:]
with h5py.File('../PlotFigure/RWR_circad_10fold_AUPR.h5') as hf:
    RWR_recall = hf['recall'][:]
    RWR_precision = hf['precision'][:]


KATZHCDA_auc = np.trapz(KATZHCDA_tpr, KATZHCDA_fpr)
KATZCPDA_auc = np.trapz(KATZCPDA_tpr, KATZCPDA_fpr)
IBNPKATZ_auc = np.trapz(IBNPKATZ_tpr , IBNPKATZ_fpr)
# BRWSP_auc = np.trapz(BRWSP_tpr , BRWSP_fpr)
CD_LNLP_auc = np.trapz(CD_LNLP_tpr, CD_LNLP_fpr)
LLCDC_auc = np.trapz(LLCDC_tpr , LLCDC_fpr)
ICIRCDA_MF_auc = np.trapz(ICIRCDA_MF_tpr , ICIRCDA_MF_fpr)
# RWR_KNN_auc = np.trapz(RWR_KNN_tpr, RWR_KNN_fpr)
# GBDTCDA_auc = np.trapz(GBDTCDA_tpr, GBDTCDA_fpr)
# DWNN_RLS_auc = np.trapz(DWNN_RLS_tpr, DWNN_RLS_fpr)
RWR_auc = np.trapz(RWR_tpr, RWR_fpr)


KATZHCDA_AUPR = np.trapz(KATZHCDA_precision, KATZHCDA_recall)
KATZCPDA_AUPR = np.trapz(KATZCPDA_precision, KATZCPDA_recall)
IBNPKATZ_AUPR = np.trapz(IBNPKATZ_precision, IBNPKATZ_recall)
# BRWSP_AUPR = np.trapz(BRWSP_precision, BRWSP_recall)
CD_LNLP_AUPR = np.trapz(CD_LNLP_precision, CD_LNLP_recall)
LLCDC_AUPR = np.trapz(LLCDC_precision, LLCDC_recall)
ICIRCDA_MF_AUPR = np.trapz(ICIRCDA_MF_precision, ICIRCDA_MF_recall)
# RWR_KNN_AUPR = np.trapz(RWR_KNN_precision, RWR_KNN_recall)
# GBDTCDA_AUPR = np.trapz(GBDTCDA_precision, GBDTCDA_recall)
# DWNN_RLS_AUPR = np.trapz(DWNN_RLS_precision, DWNN_RLS_recall)
RWR_AUPR = np.trapz(RWR_precision, RWR_recall)


plt.plot([0, 1], [0, 1], 'grey', linestyle='--')
plt.plot(RWR_fpr, RWR_tpr, 'dodgerblue', label='RWR AUC=%0.4f'% RWR_auc) #0.8667
plt.plot(KATZHCDA_fpr, KATZHCDA_tpr, 'darkorange', label='KATZHCDA AUC=%0.4f'% KATZHCDA_auc) #0.8514
plt.plot(CD_LNLP_fpr, CD_LNLP_tpr, 'hotpink', label='CD_LNLP AUC=%0.4f'% CD_LNLP_auc) #0.8380
plt.plot(LLCDC_fpr, LLCDC_tpr, 'blueviolet', label='LLCDC AUC=%0.4f'% LLCDC_auc) #0.6822
plt.plot(KATZCPDA_fpr, KATZCPDA_tpr, 'red', label='KATZCPDA AUC=%0.4f'% KATZCPDA_auc) #0.6533
plt.plot(IBNPKATZ_fpr, IBNPKATZ_tpr, 'g', label='IBNPKATZ AUC=%0.4f'% IBNPKATZ_auc)  #0.5990
plt.plot(ICIRCDA_MF_fpr, ICIRCDA_MF_tpr, 'cyan', label='ICIRCDA_MF AUC=%0.4f'% ICIRCDA_MF_auc) #0.5604
# plt.plot(RWR_KNN_fpr, RWR_KNN_tpr, 'cornflowerblue', label='RWR-KNN AUC=%0.4f'% RWR_KNN_auc)
# plt.plot(DWNN_RLS_fpr, DWNN_RLS_tpr, 'y', label='DWNN_RLS AUC=%0.4f'% DWNN_RLS_auc)
# plt.plot(BRWSP_fpr, BRWSP_tpr, 'saddlebrown', label='BRWSP AUC=%0.4f'% BRWSP_auc)
# plt.plot(GBDTCDA_fpr, GBDTCDA_tpr, 'peru', label='GBDTCDA AUC=%0.4f'% GBDTCDA_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize=8)
plt.savefig("../FinalResultPng/Figure AUC of all models dataset5 10fold.png")
plt.show()

plt.plot(KATZHCDA_recall, KATZHCDA_precision,'darkorange', label='KATZHCDA AUPR=%0.4f' % KATZHCDA_AUPR) #0.0054
plt.plot(RWR_recall, RWR_precision,'dodgerblue', label='RWR AUPR=%0.4f' % RWR_AUPR) #0.0045
plt.plot(CD_LNLP_recall, CD_LNLP_precision,'hotpink', label='CD_LNLP AUPR=%0.4f' % CD_LNLP_AUPR) #0.0033
plt.plot(LLCDC_recall, LLCDC_precision,'blueviolet', label='LLCDC AUPR=%0.4f' % LLCDC_AUPR) #0.0012
plt.plot(KATZCPDA_recall, KATZCPDA_precision,'red', label='KATZCPDA AUPR=%0.4f' % KATZCPDA_AUPR) #0.0011
plt.plot(IBNPKATZ_recall, IBNPKATZ_precision,'g', label='IBNPKATZ AUPR=%0.4f' % IBNPKATZ_AUPR) #0.0010
plt.plot(ICIRCDA_MF_recall, ICIRCDA_MF_precision,'cyan', label='ICIRCDA_MF AUPR=%0.4f' % ICIRCDA_MF_AUPR) #0.0009
# plt.plot(RWR_KNN_recall, RWR_KNN_precision,'cornflowerblue', label='RWR-KNN AUPR=%0.4f' % RWR_KNN_AUPR)
# plt.plot(DWNN_RLS_recall, DWNN_RLS_precision,'y', label='DWNN_RLS AUPR=%0.4f' % DWNN_RLS_AUPR)
# plt.plot(BRWSP_recall, BRWSP_precision,'saddlebrown', label='BRWSP AUPR=%0.4f' % BRWSP_AUPR)
# plt.plot(GBDTCDA_recall, GBDTCDA_precision, 'peru', label='GBDTCDA AUC=%0.4f'% GBDTCDA_AUPR)
plt.xlabel('Recall')
plt.ylabel('precision')
plt.legend(loc=1, fontsize = 8)
plt.savefig("../FinalResultPng/Figure AUPR of all models dataset5 10fold.png")
plt.show()


