# CompareExperiment
This is the code and the data of an underview paper named "Benchmarking computational methods for predicting circRNA-disease asssociations", which contains 10 methods and 5 collected datasets. The detail will be introduced after the paper is accepted.

# Environment Requirement
+ python == 3.6.2
+ h5py == 3.1.0
+ matplotlib == 3.3.3
+ numpy == 1.19.5
+ scipy == 1.5.4
+ pandas == 1.1.5

# Dataset
## Dataset 1 (circR2Disease)
path: CompareExperiment/Data/disease-circRNA.h5


The dataset is collected from <http://bioinfo.snnu.edu.cn/CircR2Disease/>. There are 613 known circRNA-disease associations based on 533 circRNAs and 89 diseases in this dataset.
+ CompareExperiment/Data/circRNA_list.csv records the name and the id of 533 circRNAs.
+ CompareExperiment/Data/Disease_List.csv records the name and the id of 89 disease.
+ CompareExperiment/Data/disease-circRNA.h5 records the 613 known associations.

## Dataset 2 (circR2Cancer)
path: CompareExperiment/Data/circRNA_cancer


The dataset is collected from <http://www.biobdlab.cn:8000/>. There are 647 known circRNA-disease associations based on 514 circRNAs and 62 diseases in this dataset.
+ CompareExperiment/Data/circRNA_cancer/circRNA_list.xlsx records the name and the id of 514 circRMAs.
+ CompareExperiment/Data/circRNA_cancer/disease_list.xlsx records the name and the id of 62 diseases.
+ CompareExperiment/Data/circRNA_cancer/circRNA_cancer_association.xlsx records the 647 known associations.
+ CompareExperiment/Data/circRNA_cancer/circRNA_cancer.h5 records the tuple (circRNA id, disease id).

## Dataset 3 (circRNADisease)
path: CompareExperiment/Data/circRNA_disease_from_circRNADisease


The dataset is collected from <http://cgga.org.cn:9091/circRNADisease/>. There are 331 known circRNA-disease associations based on 312 circRNAs and 40 diseases in this dataset.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/all_circRNAs.csv records the name and the id of 312 circRNAs.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/all_diseases.csv records the name and the id of 40 diseases.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/association.csv records the 331 known circRNA-disease associations.
+ CompareExperiment/Data/circRNA_disease_from_circRNADisease/association.h5 
