import os
import time

#######################################################################################################
#这里是五折
#注意更改
# denovonums = 89 #dataset1
denovonums = 62 #dataset2
# denovonums = 40 #dataset3
# denovonums = 104 #dataset4
# denovonums = 151 #dataset5
datasetname = 'dataset2'  #注意更改
foldi = 1
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
for i in range(5):
    datapath = '../Data/%s/fold_file/'%datasetname
    datasetpath = 'circRNA-disease-fold%d'%foldi
    featurepath = './Feature/5fold/%s_gcn_embedding_128_feature_fold%d.h5'%(datasetname,foldi)
    print('foldi',foldi)
    foldi +=1
    cmd = "python NGCF.py --data_path %s --dataset %s --FeaturePath %s"%(datapath,datasetpath,featurepath)

    # print(cmd)
    os.system(cmd)
print("5fold finish")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#######################################################################################################

#######################################################################################################
#这里是十折
foldi = 1
# datasetname = 'dataset2'
for i in range(10):
    datapath = '../Data/%s/fold10/'%datasetname
    datasetpath = 'circRNA-disease-fold%d'%foldi
    featurepath = './Feature/5fold/%s_gcn_embedding_128_feature_fold%d.h5'%(datasetname,foldi)
    print('foldi',foldi)
    foldi +=1
    cmd = "python NGCF.py --data_path %s --dataset %s --FeaturePath %s"%(datapath,datasetpath,featurepath)

    # print(cmd)
    os.system(cmd)
print("10fold finish")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#######################################################################################################

#######################################################################################################
#这里是denovo
foldi = 1
# datasetname = 'dataset2'
for i in range(denovonums):
    datapath = '../Data/%s/denovo/'%datasetname
    datasetpath = 'circRNA-disease-fold%d'%foldi
    featurepath = './Feature/5fold/%s_gcn_embedding_128_feature_fold%d.h5'%(datasetname,foldi)
    print('foldi',foldi)
    foldi +=1
    cmd = "python NGCF.py --data_path %s --dataset %s --FeaturePath %s"%(datapath,datasetpath,featurepath)

    # print(cmd)
    os.system(cmd)
print("denovo finish")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#######################################################################################################