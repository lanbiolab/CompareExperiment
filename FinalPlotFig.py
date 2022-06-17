'''
@Author: DongYi
@Date: 2020/10/25
@Description: 这部分代码是将所有的曲线图整合绘制到一张大图上
这里要涉及到用一些颜色进行调色
'''
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

# # 先绘制AUC图
# with h5py.File('./PlotFigure/RWR_AUC.h5') as f:
#     RWR_tpr = f['tpr'][:]
#     RWR_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/KATZHCDA_AUC.h5') as f:
#     KATZHCDA_tpr = f['tpr'][:]
#     KATZHCDA_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/CD-LNLP_AUC.h5') as f:
#     CD_LNLP_tpr = f['tpr'][:]
#     CD_LNLP_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/KATZCPDA_AUC.h5') as f:
#     KATZCPDA_tpr = f['tpr'][:]
#     KATZCPDA_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/DWNN-RLS_AUC.h5') as f:
#     DWNN_RLS_tpr = f['tpr'][:]
#     DWNN_RLS_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/IBNPKATZ_AUC.h5') as f:
#     IBNPKATZ_tpr = f['tpr'][:]
#     IBNPKATZ_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/LLCDC_AUC.h5') as f:
#     LLCDC_tpr = f['tpr'][:]
#     LLCDC_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/iCircDA-MF_AUC.h5') as f:
#     ICIRCDA_MF_tpr = f['tpr'][:]
#     ICIRCDA_MF_fpr = f['fpr'][:]
# with h5py.File('./PlotFigure/BRWSP_AUC.h5') as f:
#     BRWSP_tpr = f['tpr'][:]
#     BRWSP_fpr = f['fpr'][:]
# # with h5py.File('./PlotFigure/GBDTCDA_AUC.h5') as f:
# #     GBDTCDA_tpr = f['tpr'][:]
# #     GBDTCDA_fpr = f['fpr'][:]
#
#
# RWR_auc = np.trapz(RWR_tpr, RWR_fpr)
# KATZHCDA_auc = np.trapz(KATZHCDA_tpr, KATZHCDA_fpr)
# CD_LNLP_auc = np.trapz(CD_LNLP_tpr, CD_LNLP_fpr)
# KATZCPDA_auc = np.trapz(KATZCPDA_tpr, KATZCPDA_fpr)
# DWNN_RLS_auc = np.trapz(DWNN_RLS_tpr, DWNN_RLS_fpr)
# IBNPKATZ_auc = np.trapz(IBNPKATZ_tpr , IBNPKATZ_fpr)
# LLCDC_auc = np.trapz(LLCDC_tpr , LLCDC_fpr)
# ICIRCDA_MF_auc = np.trapz(ICIRCDA_MF_tpr , ICIRCDA_MF_fpr)
# BRWSP_auc = np.trapz(BRWSP_tpr , BRWSP_fpr)
#
# plt.plot([0, 1], [0, 1], 'grey', linestyle='--')
# plt.plot(RWR_fpr, RWR_tpr, 'dodgerblue', label='RWR AUC=%0.4f'% RWR_auc)
# plt.plot(KATZHCDA_fpr, KATZHCDA_tpr, 'darkorange', label='KATZHCDA AUC=%0.4f'% KATZHCDA_auc)
# plt.plot(CD_LNLP_fpr, CD_LNLP_tpr, 'red', label='CD_LNLP AUC=%0.4f'% CD_LNLP_auc)
# plt.plot(KATZCPDA_fpr, KATZCPDA_tpr, 'hotpink', label='KATZCPDA AUC=%0.4f'% KATZCPDA_auc)
# plt.plot(DWNN_RLS_fpr, DWNN_RLS_tpr, 'y', label='DWNN_RLS AUC=%0.4f'% DWNN_RLS_auc)
# plt.plot(IBNPKATZ_fpr, IBNPKATZ_tpr, 'g', label='IBNPKATZ AUC=%0.4f'% IBNPKATZ_auc)
# plt.plot(LLCDC_fpr, LLCDC_tpr, 'blueviolet', label='LLCDC AUC=%0.4f'% LLCDC_auc)
# plt.plot(ICIRCDA_MF_fpr, ICIRCDA_MF_tpr, 'cyan', label='ICIRCDA_MF AUC=%0.4f'% ICIRCDA_MF_auc)
# plt.plot(BRWSP_fpr, BRWSP_tpr, 'saddlebrown', label='BRWSP AUC=%0.4f'% BRWSP_auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc=0)
# plt.savefig("./FinalResultPng/Figure AUC of all models.png")
# plt.show()

# 接下来是AUPR图
# with h5py.File('./PlotFigure/RWR_AUPR.h5') as hf:
#     RWR_recall = hf['recall'][:]
#     RWR_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/KATZHCDA_AUPR.h5') as hf:
#     KATZHCDA_recall = hf['recall'][:]
#     KATZHCDA_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/CD-LNLP_AUPR.h5') as hf:
#     CD_LNLP_recall = hf['recall'][:]
#     CD_LNLP_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/KATZCPDA_AUPR.h5') as hf:
#     KATZCPDA_recall = hf['recall'][:]
#     KATZCPDA_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/DWNN-RLS_AUPR.h5') as hf:
#     DWNN_RLS_recall = hf['recall'][:]
#     DWNN_RLS_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/IBNPKATZ_AUPR.h5') as hf:
#     IBNPKATZ_recall = hf['recall'][:]
#     IBNPKATZ_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/LLCDC_AUPR.h5') as hf:
#     LLCDC_recall = hf['recall'][:]
#     LLCDC_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/iCircDA-MF_AUPR.h5') as hf:
#     ICIRCDA_MF_recall = hf['recall'][:]
#     ICIRCDA_MF_precision = hf['precision'][:]
# with h5py.File('./PlotFigure/BRWSP_AUPR.h5') as hf:
#     BRWSP_recall = hf['recall'][:]
#     BRWSP_precision = hf['precision'][:]
#
# RWR_AUPR = np.trapz(RWR_precision, RWR_recall)
# KATZHCDA_AUPR = np.trapz(KATZHCDA_precision, KATZHCDA_recall)
# CD_LNLP_AUPR = np.trapz(CD_LNLP_precision, CD_LNLP_recall)
# KATZCPDA_AUPR = np.trapz(KATZCPDA_precision, KATZCPDA_recall)
# DWNN_RLS_AUPR = np.trapz(DWNN_RLS_precision, DWNN_RLS_recall)
# IBNPKATZ_AUPR = np.trapz(IBNPKATZ_precision, IBNPKATZ_recall)
# LLCDC_AUPR = np.trapz(LLCDC_precision, LLCDC_recall)
# ICIRCDA_MF_AUPR = np.trapz(ICIRCDA_MF_precision, ICIRCDA_MF_recall)
# BRWSP_AUPR = np.trapz(BRWSP_precision, BRWSP_recall)
#
# plt.plot(KATZHCDA_recall, KATZHCDA_precision,'darkorange', label='KATZHCDA AUPR=%0.4f' % KATZHCDA_AUPR)
# plt.plot(RWR_recall, RWR_precision,'dodgerblue', label='RWR AUPR=%0.4f' % RWR_AUPR)
# plt.plot(CD_LNLP_recall, CD_LNLP_precision,'red', label='CD_LNLP AUPR=%0.4f' % CD_LNLP_AUPR)
# plt.plot(KATZCPDA_recall, KATZCPDA_precision,'hotpink', label='KATZCPDA AUPR=%0.4f' % KATZCPDA_AUPR)
# plt.plot(DWNN_RLS_recall, DWNN_RLS_precision,'y', label='DWNN_RLS AUPR=%0.4f' % DWNN_RLS_AUPR)
# plt.plot(IBNPKATZ_recall, IBNPKATZ_precision,'g', label='IBNPKATZ AUPR=%0.4f' % IBNPKATZ_AUPR)
# plt.plot(LLCDC_recall, LLCDC_precision,'blueviolet', label='LLCDC AUPR=%0.4f' % LLCDC_AUPR)
# plt.plot(BRWSP_recall, BRWSP_precision,'saddlebrown', label='BRWSP AUPR=%0.4f' % BRWSP_AUPR)
# plt.plot(ICIRCDA_MF_recall, ICIRCDA_MF_precision,'cyan', label='ICIRCDA_MF AUPR=%0.4f' % ICIRCDA_MF_AUPR)
#
# plt.xlabel('Recall')
# plt.ylabel('precision')
# plt.legend(loc=0)
# plt.savefig("./FinalResultPng/Figure AUPR of all models.png")
# plt.show()

#======================================================================================================================#

# 绘制denovo图
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
accuracy_list = []
F1_list = []

path = './denovo/iCircDA-MF_circ2Traits_denovo/results'
for file in os.listdir(path):
    file_path = os.path.join(path,file)
    with h5py.File(file_path,'r') as hf:
        tpr_list += (hf['tpr_arr'][:]).tolist()
        fpr_list += (hf['fpr_arr'][:]).tolist()
        recall_list += (hf['recall_arr'][:]).tolist()
        precision_list += (hf['precision_arr'][:]).tolist()
        accuracy_list += (hf['accuracy_arr'][:]).tolist()
        F1_list += (hf['F1_arr'][:]).tolist()

tpr_arr = np.array(tpr_list)
fpr_arr = np.array(fpr_list)
recall_arr = np.array(recall_list)
precision_arr = np.array(precision_list)
accuracy_arr = np.array(accuracy_list)
F1_arr = np.array(F1_list)

mean_denovo_tpr = np.mean(tpr_arr, axis=0)  # axis=0
mean_denovo_fpr = np.mean(fpr_arr, axis=0)
mean_denovo_recall = np.mean(recall_arr, axis=0)
mean_denovo_precision = np.mean(precision_arr, axis=0)
mean_denovo_accuracy = np.mean(accuracy_arr, axis=0)
# 计算此次五折的平均评价指标数值
mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_denovo_tpr, mean_denovo_fpr)
AUPR = np.trapz(mean_denovo_precision, mean_denovo_recall)
print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))

# 存储tpr，fpr,recall,precision
with h5py.File('./PlotFigure/iCircDA_MF_circ2Traits_denovo_AUC.h5', 'w') as hf:
    hf['fpr'] = mean_denovo_fpr
    hf['tpr'] = mean_denovo_tpr
with h5py.File('./PlotFigure/iCircDA_MF_circ2Traits_denovo_AUPR.h5', 'w') as h:
    h['recall'] = mean_denovo_recall
    h['precision'] = mean_denovo_precision

plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("./FinalResultPng/roc-iCircDA_MF_circ2Traits_denovo.png")
print("runtime over, now is :")
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()
