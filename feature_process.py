"""
This module provides functions for processing features for an autoencoder model
used in gearbox fault diagnosis. It includes functions for window sampling, 
data concatenation, normalization, and separating normal and anomalous data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Use window to sample the data, and decreasing the data size.
# For autoencoder using with normal&anomaly label.
# User can update the different features here.
def window_feature_autoencoder(df_data, normal_label, win_size, in_dim, out_dim):
    """
    Extracts features from the input dataframe using a sliding window approach and returns the features along with labels.
    Returns:
    tuple: A tuple containing:
        - data_feature (np.ndarray): Extracted features with shape (num_windows, out_dim).
        - label (np.ndarray): Array of labels with shape (num_windows,).
    """

    data_feature = np.zeros([int(np.round(len(df_data) / win_size)), out_dim])
    for j in range(0, in_dim):
        feature_idx = 0
        for i in range((win_size), len(df_data), win_size):
            # data_feature[feature_idx,j+4]=  df_data.iloc[i-100:i,j].skew()
            data_feature[feature_idx, j] = df_data.iloc[i - win_size : i, j].abs().sum()
            data_feature[feature_idx, j + 4] = df_data.iloc[i - win_size : i, j].std()
            data_feature[feature_idx, j + 8] = df_data.iloc[i - win_size : i, j].max()  # 0.8159
            data_feature[feature_idx, j + 12] = df_data.iloc[i - win_size : i, j].kurt()  # 0.8172

            feature_idx = feature_idx + 1

    label = normal_label * np.ones([int(np.round(len(df_data) / win_size))])
    return data_feature, label


# combine the normal & anomaly data into 1 dataframe and split them into train & test
def concatenate_data(data_healthy, data_broken, lable_healthy, lable_broken, testsize, random_seed):
    """
    Concatenates healthy and broken data along with their labels, then splits the combined data into training and testing sets.
    tuple: A tuple containing four elements:
        - train_data (numpy.ndarray): Training data samples.
        - test_data (numpy.ndarray): Testing data samples.
        - train_labels (numpy.ndarray): Labels for the training data.
        - test_labels (numpy.ndarray): Labels for the testing data.
    """
    data = np.concatenate([data_healthy, data_broken], axis=0)
    labels = np.concatenate([lable_healthy, lable_broken], axis=0)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=testsize, random_state=random_seed)

    return train_data, test_data, train_labels, test_labels


# normalize data with Max & Min value of train_data
# return the normalized train and test data
def normalize_data_maxmin(train_data, test_data):
    """
    Normalize the training and testing data using min-max normalization.
    This function scales the input data to a range between 0 and 1 based on the minimum and maximum values
    found in the training data. The same scaling is applied to the test data.
    Args:
        train_data (tf.Tensor): The training data to be normalized.
        test_data (tf.Tensor): The testing data to be normalized.
    Returns:
        tuple: A tuple containing the normalized training data and testing data as tf.float32 tensors.
    """
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    print(max_val, min_val)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    return train_data, test_data


# return normal & anomalous from an input dataset.
def normal_anomalous_distb(ts_data, labels):
    """
    Separates time series data into normal and anomalous subsets based on provided labels.
    Args:
    ts_data (array-like): The time series data to be separated.
    labels (array-like): Boolean labels indicating normal (True) and anomalous (False) data points.
    Returns:
    tuple: A tuple containing two elements:
        - normal_data (array-like): The subset of ts_data where labels are True.
        - anomalous_data (array-like): The subset of ts_data where labels are False.
    """
    labels = labels.astype(bool)
    normal_data = ts_data[labels]
    anomalous_data = ts_data[~labels]

    return normal_data, anomalous_data
