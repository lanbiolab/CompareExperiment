# CompareExperiment
Accumulating evidences demonstrate that circular RNA (circRNA) plays an important role in human diseases. Identification of circRNA-disease associations can help for the diagnosis of human diseases, while the traditional method based on biological experiments is time-consuming. In order to address the limitation, a series of computational methods have been proposed in recent years. However, few works have summarized these methods or compared the performance of them. In this paper, we divided the existing methods into three categories: information propagation, traditional machine learning and deep learning. Then, the baseline methods in each category are introduced in detail. Further, 5 different datasets are collected, and 14 representative methods of each category are selected and compared in the 5-fold, 10-fold cross-validation and the de novo experiment. In order to further evaluate the effectiveness of these methods, six common cancers are selected to compare the number of correctly identified circRNA-disease associations in the top-10, top-20, top-50, top-100 and top-200. In addition, according to the results, the observation about the robustness and the character of these methods are concluded. Finally, the future directions and challenges are discussed.

Author: Wei Lan. Yi Dong. Hongyu Zhang. Chunling Li. Qingfeng Chen. Jin Liu. Jianxin Wang. Yi-Ping Phoebe Chen.

Paper: https://academic.oup.com/bib/article-abstract/24/1/bbac613/6972300?redirectedFrom=fulltext

# Citation
If you want to use our codes and datasets in your research, please cite:
```
@article{10.1093/bib/bbac613,
    author = {Lan, Wei and Dong, Yi and Zhang, Hongyu and Li, Chunling and Chen, Qingfeng and Liu, Jin and Wang, Jianxin and Chen, Yi-Ping Phoebe},
    title = "{Benchmarking of computational methods for predicting circRNA-disease associations}",
    journal = {Briefings in Bioinformatics},
    volume = {24},
    number = {1},
    year = {2023},
    month = {01},
    issn = {1477-4054},
    doi = {10.1093/bib/bbac613},
    url = {https://doi.org/10.1093/bib/bbac613},
    note = {bbac613},
    eprint = {https://academic.oup.com/bib/article-pdf/24/1/bbac613/48781859/bbac613.pdf},
}
```

# Environment Requirement
+ python == 3.6.2
+ h5py == 3.1.0
+ matplotlib == 3.3.3
+ numpy == 1.19.5
+ scipy == 1.5.4
+ pandas == 1.1.5

# Dataset
## Dataset 1 (circR2Disease)
data path: CompareExperiment/Data/disease-circRNA.h5


The dataset is collected from <http://bioinfo.snnu.edu.cn/CircR2Disease/>. There are 613 known circRNA-disease associations based on 533 circRNAs and 89 diseases in this dataset.
+ CompareExperiment/Data/circRNA_list.csv records the name and the id of 533 circRNAs.
+ CompareExperiment/Data/Disease_List.csv records the name and the id of 89 disease.
+ CompareExperiment/Data/disease-circRNA.h5 records the 613 known associations.
+ CompareExperiment/Data/disease-circRNA.h5 records the known circRNA-disease tuple (circRNA id, disease id).

## Dataset 2 (circR2Cancer)
data path: CompareExperiment/Data/circRNA_cancer


The dataset is collected from <http://www.biobdlab.cn:8000/>. There are 647 known circRNA-disease associations based on 514 circRNAs and 62 diseases in this dataset.
+ CompareExperiment/Data/circRNA_cancer/circRNA_list.xlsx records the name and the id of 514 circRMAs.
+ CompareExperiment/Data/circRNA_cancer/disease_list.xlsx records the name and the id of 62 diseases.
+ CompareExperiment/Data/circRNA_cancer/circRNA_cancer_association.xlsx records the 647 known associations.
+ CompareExperiment/Data/circRNA_cancer/circRNA_cancer.h5 records the known circRNA-disease tuple (circRNA id, disease id).

## Dataset 3 (circRNADisease)
data path: CompareExperiment/Data/circRNA_disease_from_circRNADisease


The dataset is collected from <http://cgga.org.cn:9091/circRNADisease/>. There are 331 known circRNA-disease associations based on 312 circRNAs and 40 diseases in this dataset.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/all_circRNAs.csv records the name and the id of 312 circRNAs.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/all_diseases.csv records the name and the id of 40 diseases.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/association.csv records the 331 known circRNA-disease associations.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/association.h5 records the known circRNA-disease tuple (circRNA id, disease id).

## Dataset 4 (circ2Traits)
data path: CompareExperiment/Data/circ2Traits


The dataset is collected from <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3857533/>. There are 37660 known circRNA-disease associations based on 923 circRNAs and 104 diseases in this dataset.
+ CompareExperiment/Data/circ2Traits/circRNA_list.csv records the name and the id of 923 circRNAs.
+ CompareExperiment/Data/circ2Traits/disease_list.csv records the name and the id of 104 diseases.
+ CompareExperiment/Data/circ2Traits/circRNA_disease.csv records the 37660 known circRNA-disease associations.
+ CompareExperiment/Data/circ2Traits/circRNA_disease.h5 records the known circRNA-disease tuple (circRNA id, disease id).

## Dataset 5 (circad)
data path: CompareExperiment/Data/circad


The dataset is collected from <https://clingen.igib.res.in/circad/>. There are 1369 known circRNA-disease associations based on 1265 circRNAs and 151 diseases in this dataset.
+ CompareExperiment/Data/circad/circrna_list.xls records the name and the id of 1265 circRNAs.
+ CompareExperiment/Data/circad/disease_list.xls records the name and the id of 151 diseases.
+ CompareExperiment/Data/circad/circrna_disease.h5 records the records the known circRNA-disease tuple (circRNA id, disease id).

# Methods
+ KATZHCDA: This is a baseline method based on KATZ. Fan et al. proposed KATZHCDA to predict circRNA-disease association.


  origin paper: [KATZHCDA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6299360/)


  method path: CompareExperiment/KATZHCDA.py

+ KATZCPDA: Deng et al. proposed this method. They combine the inferred circRNA-disease association matrix with origin circRNA-disease association matrix. Then the KATZ is used on the fusion matrix to predict circRNA-disease associations.


  origin paper: [KATZCPDA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6610109/)


  method path: CompareExperiment/KATZCPDA.py
  
+ IBNPKATZ: Zhao et al. integrated the bipartite network projection algorithm with KATZ to predict circRNA-disease associations.


  origin paper: [IBNPKATZ](https://ieeexplore.ieee.org/document/8735932)


  method path: CompareExperiment/IBNPKATZ.py

+ CD-LNLP: Zhang et al. proposed CD-LNLP based on linear neighborhood label propagation to predict circRNA-disease associations.


  origin paper: [CD-LNLP](https://ieeexplore.ieee.org/document/8731942)


  method path: CompareExperiment/CD-LNLP.py
  
+ LLCDC: Ge et al. developed Locality-Constrained Linear Coding (LLCDC) to identify circRNA-disease associations.


  origin paper: [LLCDC](https://pubmed.ncbi.nlm.nih.gov/31394170/)
  
  
  method path: CompareExperiment/LLCDC.py
  
+ RWR: This is a baseline method based on Random Walk with Restart algorithm.


  origin paper: [RWR](https://ieeexplore.ieee.org/abstract/document/9073607)
  
  
  method path:  CompareExperiment/RWR.py

+ RWRKNN: Lei et al. combined the RWR and KNN to predict circRNA-disease associations.


  origin paper: [RWRKNN](https://www.nature.com/articles/s41598-020-59040-0)
  
  
  method path: CompareExperiment/RWRKNN.py
  
+ iCircDA-MF: Wei et al. proposed iCircDA-MF to predict circRNA-disease association based on Matrix Factorization.


  origin paper: [iCircDA-MF](https://pubmed.ncbi.nlm.nih.gov/32241268/#:~:text=Prediction%20of%20circRNA-disease%20associations%20based%20on%20inductive%20matrix,the%20field%20and%20follow-up%20investigations%20by%20biomedical%20researchers.)
  
  
  method path: CompareExperiment/iCircDA-MF.py
  
+ DWNN-RLS: Yan et al. proposed a computational method (DWNN-RLS) to predict circRNA-disease associations. The Kronecker product kernel based regularized least squares (RLS-Kron) is used to calculate the affinity scores of circRNA-disease associations.


  origin paper: [DWNN-RLS](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2522-6)
  
  
  method path: CompareExperiment/DWNN-RLS.py
  
+ DMFCDA: Lu et al. proposed a computational method (DMFCDA) to infer circRNA-disease associations based on neural network.


  origin paper: [DMFCDA](https://ieeexplore.ieee.org/document/9107417)
  
  
  method path: CompareExperiment/DMFCDA.py
  
+ GMNN2CD：Niu et al. proposed a computational method (GMNN2CD) to predict circRNA-disease associations based on variational inference and graph Markov neural networks.


  origin paper: [GMNN2CD](https://academic.oup.com/bioinformatics/article-abstract/38/8/2246/6528308)


  method path: CompareExperiment/GMNN2CD.py

+ IGNSCDA： Lan et al. proposed a computational method (IGNSCDA) to predict associations between circRNAs and diseases based on improved graph convolutional network and negative sampling method.


  origin paper: [IGNSCDA](https://ieeexplore.ieee.org/abstract/document/9535297?casa_token=hnXAxnRzEhUAAAAA:94GJioB8yANzNX7wTLpDfEGyxpNyUa8gjP2fsijawRW5_aB9tb7MNNNR7DRlDcfI90we5JRKs09nuA)
  
  
  method path: CompareExperiment/NGCF
  
  
  This is a sub-project in CompareExperiment, you can run CompareExperiment/NGCF/NGCF.py in this project to generate the embeddings of circRNAs and diseases, then the embedding will be saved as .h5 file in CompareExperiment/NGCF/Feature/5fold (10fold or denovo). The more details about IGNSCDA: https://github.com/lanbiolab/IGNSCDA
  
  
+ RNMFLP: Peng et al. proposed a computational method (RNMFLP) to infer circRNA-disease associations based on robust nonnegative matrix factorization and label propagation.


  origin paper: [RNMFLP](https://academic.oup.com/bib/article-abstract/23/5/bbac155/6582881)
  
  
  method path: CompareExperiment/RNMFLP main
  
  
  This is a sub-project written in matlab. We have run the project on our 5 common datasets as the origin paper recommends. The more details please refer: :https://github.com/biohnuster/RNMFLP

+ AE-RF: K. Deepthi et al. proposed a computational method (AE-RF) to predict circRNA-disease associations based on deep autoencoder-based classification.


    origin paper: [AE-RF](https://link.springer.com/article/10.1007/s40291-020-00499-y)
  
  
    method path: you can run CompareExperiment/AE-RF main/AE-RF.py in this project.
  
  
# Experiment
## 5-fold, 10-fold cross validation
  You can find the code as follows:
```
split = math.ceil(len(one_list) / 5)
```
  if you change '5' to '10', the 5-fold cross validation experiment will be switched to 10-fold cross validation experiment.

## de-novo experiment
  The path of each method in de-novo experiment will be introduced. For some complicate methods, we divide the dieases of these methods into multiple groups to save the training time. 
+ KATZHCDA: CompareExperiment/denovo/KATZHCDA_circ2Traits_denovo, CompareExperiment/denovo/KATZHCDA_circad_denovo, CompareExperiment/KATZHCDA_denovo.py
+ KATZCPDA: CompareExperiment/denovo/KATZCPDA_circ2Traits_denovo, CompareExperiment/denovo/KATZCPDA_circad_denovo, CompareExperiment/KATZCPDA_denovo.py
+ IBNPKATZ: CompareExperiment/denovo/IBNPKATZ_circ2Traits_denovo, CompareExperiment/denovo/IBNPKATZ_circad_denovo, CompareExperiment/IBNPKATZ_denovo.py
+ CD-LNLP: CompareExperiment/CD-LNLP_denovo.py
+ LLCDC: CompareExperiment/denovo/LLCDC_circ2Traits_denovo, CompareExperiment/denovo/LLCDC_circad_denovo, CompareExperiment/LLCDC_denovo.py
+ RWR: CompareExperiment/denovo/RWR_circ2Traits_denovo, CompareExperiment/denovo/RWR_circad_denovo, CompareExperiment/RWR_denovo.py
+ RWRKNN: CompareExperiment/denovo/RWRKNN_circ2Traits_denovo, CompareExperiment/denovo/RWRKNN_circad_denovo, CompareExperiment/RWRKNN_denovo.py
+ iCircDA-MF: CompareExperiment/denovo/iCircDA-MF_circ2Traits_denovo, CompareExperiment/denovo/iCircDA-MF_circad_denovo, CompareExperiment/iCircDA-MF_denovo.py
+ DWNN-RLS: CompareExperiment/denovo/iCircDA-MF_circ2Traits_denovo, CompareExperiment/denovo/iCircDA-MF_circad_denovo, CompareExperiment/iCircDA-MF_denovo.py
+ DMFCDA: CompareExperiment/DMFCDA_circ2Traits_denovo, CompareExperiment/DMFCDA_circad_denovo, CompareExperiment/DMFCDA_circR2Disease_denovo, CompareExperiment/DMFCDA_circRNA_cancer_denovo, CompareExperiment/DMFCDA_small_circRNADisease_denovo

# Other files
## plot_fig
  The files in 'CompareExperiment/plot_fig' package aims to plot the experiment results (AUC, AUPR) in each dataset.
  For example, 'plot_fig_dataset1.py' aims to plot the 5-fold experiment results (AUC, AUPR). 
  
  
  'plot_fig_dataset1_10fold.py' aims to plot the 10-fold experiment results (AUC, AUPR).
  
  
  'plot_fig_dataset1_denovo.py' aims to plot the de-novo experiment results (AUC, AUPR).
  
## PlotFigure
  '...AUC.h5' file contains the tpr and fpr data, which can be used to calculate the AUC value and plot ROC curve. Similarly, '...AUPR.h5' file saves the precision and recall data, whcih can be used to calculate the AUPR value and plot PR curve.
  
## FinalResultPng
  This file records the AUC and AUPR figures in each experiment.

# Questions
If you have any problems or find mistakes in this code, please contact with me: Yi Dong: dongyi6463@163.com
