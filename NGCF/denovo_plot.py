import h5py
import numpy as np
import matplotlib.pyplot as plt

import time
import os

# 绘制denovo图
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
accuracy_list = []
F1_list = []

# path = './denovo/iCircDA-MF_circ2Traits_denovo/results'
path = 'D:/pycharm社区版（现在电脑上用的的不是这个版本，可以删）/IGNSCDA结果记录20220803/denovo/dataset5'
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
with h5py.File('./PlotFigure/IGNSCDA_dataset5_denovo_AUC.h5', 'w') as hf:
    hf['fpr'] = mean_denovo_fpr
    hf['tpr'] = mean_denovo_tpr
with h5py.File('./PlotFigure/IGNSCDA_dataset5_denovo_AUPR.h5', 'w') as h:
    h['recall'] = mean_denovo_recall
    h['precision'] = mean_denovo_precision

plt.plot(mean_denovo_fpr, mean_denovo_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("./FinalResultPng/IGNSCDA_dataset5_denovo_denovo.png")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()
