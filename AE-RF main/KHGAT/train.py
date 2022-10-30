import tensorflow as tf
import h5py

def load_data():
    with h5py.File('lncRNA_disease_Associations.h5', 'r') as hf:
        circrna_disease_matrix = hf['rating'][:]
    return circrna_disease_matrix
