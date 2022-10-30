import numpy as np
import matplotlib.pyplot as plt
import h5py
def load_data():
    fpr = np.loadtxt(r'E:\第四学期学习\综述实验\数据\RNMFLP\五折和十折\data5_5_fold_fpr.txt')
    tpr = np.loadtxt(r'E:\第四学期学习\综述实验\数据\RNMFLP\五折和十折\data5_5_fold_tpr.txt')
    recall = np.loadtxt(r'E:\第四学期学习\综述实验\数据\RNMFLP\五折和十折\data5_5_fold_recall.txt')
    precision = np.loadtxt(r'E:\第四学期学习\综述实验\数据\RNMFLP\五折和十折\data5_5_fold_precision.txt')
    roc_auc = np.trapz(tpr, fpr)
    aupr = np.trapz(precision, recall)
    # mean_cross_recall = np.mean(recall, axis=0)
    # mean_cross_precision = np.mean(precision, axis=0)
    # mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
    # 计算此次五折的平均评价指标数值
    # mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision, axis=1), axis=0)
    # plt.plot(fpr, tpr, label='mean ROC=%0.4f' % roc_auc)
    print(roc_auc, aupr, mean_precision, mean_recall)
    print()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=0)
    # plt.show()
def read_h5():
    with h5py.File(r'E:\pythonProject\KHGAT\data\associations5_5_auc.h5') as hf:
        fpr = hf['fpr']
        tpr = hf['tpr']
        print(fpr)
        for key in hf.keys():
            # print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
            print(hf[key], key, hf[key].name)
    with h5py.File(r'E:\pythonProject\KHGAT\data\associations5_5_aupr.h5') as hf:
        recall = hf['/recall']
        precision = hf['/precision']
    roc_auc = np.trapz(tpr, fpr)
    aupr = np.trapz(precision, recall)
    print(roc_auc, aupr)

if __name__=="__main__":
    # load_data()
    read_h5()