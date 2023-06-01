import numpy as np 
import pandas as pd
import os
from scipy.stats import norm, kurtosis, skew
from sklearn.model_selection import train_test_split
import tensorflow as tf

#class feature_process:

### Use window to sample the data, and decreasing the data size. ###
### For autoencoder using with normal&anomaly label.             ###
### User can update the different features here.                 ###
def window_feature_autoencoder(df_data, normal_label, win_size, in_dim, out_dim):

    Data_feature = np.zeros([int(np.round(len(df_data)/win_size)), out_dim])
    for j in range(0, in_dim):
        feature_idx = 0  
        for i in range((win_size), len(df_data), win_size):
            #Data_feature[feature_idx,j+4]=  df_data.iloc[i-100:i,j].skew()
            Data_feature[feature_idx,j]    = df_data.iloc[i-win_size:i,j].abs().sum()
            Data_feature[feature_idx,j+4]  = df_data.iloc[i-win_size:i,j].std()
            Data_feature[feature_idx,j+8]  = df_data.iloc[i-win_size:i,j].max() ##0.8159
            Data_feature[feature_idx,j+12] = df_data.iloc[i-win_size:i,j].kurt() ##0.8172
            
            
            feature_idx = feature_idx + 1      
    
    Label = normal_label * np.ones([int(np.round(len(df_data)/win_size))])
    return Data_feature, Label

### combine the normal & anomaly data into 1 dataframe and split them into train & test ###
def concatenate_data(Data_healthy,Data_broken,Lable_healthy,Lable_broken, testSize, randomSeed):
    data   = np.concatenate([Data_healthy,Data_broken], axis =0)
    labels = np.concatenate([Lable_healthy,Lable_broken], axis =0)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size = testSize, random_state = randomSeed)
    
    return train_data, test_data, train_labels, test_labels

### normalize data with Max & Min value of train_data ###
### return the normalized train and test data         ###
def normalize_data_maxmin(train_data, test_data):
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    print(max_val, min_val)
    
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    return train_data, test_data

### return normal & anomalous from an input dataset. ###
def normal_anomalous_distb(ts_data, labels):
    labels = labels.astype(bool)
    normal_data = ts_data[labels]
    anomalous_data = ts_data[~labels]
    
    return normal_data, anomalous_data