
'''
This module provides functions to load and process data from CSV files for gearbox fault diagnosis.
Functions:
    test():
    addcol_load(x, frame):
    addcol_fault(anomaly, frame):
    create_df_from_raw(pth, anomaly, *funcs):
    create_test_train_df_from_raw(pth, anomaly, test_len, win_size, *funcs):
'''
import os
import numpy as np
import pandas as pd

# class load_data:
def test():
    """
    A simple test function that prints 'test' to the console.
    """
    print("test")


# User decide functions for different data's format
def addcol_load(x, frame):
    """
    Adds a column named 'load' to the given DataFrame with values calculated as 10 times the input x.
    Parameters:
    x (float): The multiplier for the 'load' column values.
    frame (pandas.DataFrame): The DataFrame to which the 'load' column will be added.
    Returns:
    pandas.DataFrame: The DataFrame with the added 'load' column.
    """
    frame["load"] = 10 * x * np.ones((len(frame), 1))
    return frame


def addcol_fault(anomaly, frame):
    """
    Adds a 'fault' column to the given DataFrame.
    Parameters:
    anomaly (int or float): The value to be assigned to the 'fault' column.
    frame (pandas.DataFrame): The DataFrame to which the 'fault' column will be added.
    Returns:
    pandas.DataFrame: The DataFrame with the added 'fault' column.
    """
    frame["fault"] = anomaly * np.ones((len(frame), 1))
    return frame


# Pull data from pth, and label with normal or anomaly
# Also have adding columns with customized function
def create_df_from_raw(pth, anomaly, *funcs):
    """
    Creates a DataFrame from raw CSV files in the specified directory.
    Parameters:
    pth (str): The path to the directory containing the CSV files.
    anomaly (any): A parameter to be passed to the second function in funcs.
    *funcs (tuple): A tuple containing two functions. The first function takes
                    an integer and a DataFrame as arguments and returns a DataFrame.
                    The second function takes the anomaly parameter and a DataFrame
                    as arguments and returns a DataFrame.
    Returns:
    pd.DataFrame: A concatenated DataFrame created from the processed CSV files.
    """

    file_list = []
    for file in os.listdir(pth):
        file_list.append(os.path.join(pth, file))
    file_list.sort()

    _data_list = []
    _df = pd.DataFrame(_data_list)
    for x in range(0, 10):
        frame = pd.read_csv(file_list[x])

        frame = funcs[0][0](x, frame)
        frame = funcs[0][1](anomaly, frame)

        _df = pd.concat([_df, frame], axis=0, ignore_index=True)

    return _df


# get data for MCU inference test latter, and this data will not be train.
def create_test_train_df_from_raw(pth, anomaly, test_len, win_size, *funcs):
    """
    Creates training and testing DataFrames from raw data files.
    Parameters:
    pth (str): The path to the directory containing the raw data files.
    anomaly (any): Anomaly parameter to be passed to the second function in funcs.
    test_len (int): The length of the test set in terms of number of windows.
    win_size (int): The size of each window.
    *funcs (tuple): A tuple containing two functions. The first function processes the data based on the file index,
                    and the second function processes the data based on the anomaly parameter.
    Returns:
    tuple: A tuple containing two DataFrames. The first DataFrame is the training set and the second DataFrame is the test set.
    """

    file_list = []
    for file in os.listdir(pth):
        file_list.append(os.path.join(pth, file))
    file_list.sort()

    _data_list = []
    _df = pd.DataFrame(_data_list)
    _df_test = pd.DataFrame(_data_list)
    for x in range(0, 10):
        frame = pd.read_csv(file_list[x])

        frame = funcs[0][0](x, frame)
        frame = funcs[0][1](anomaly, frame)
        # print(frame.shape[0], math.floor(frame.shape[0]/300))

        _df_test = pd.concat([_df_test, frame[0 : test_len * win_size]], axis=0, ignore_index=True)
        _df = pd.concat([_df, frame[test_len * win_size :]], axis=0, ignore_index=True)

    return _df, _df_test
