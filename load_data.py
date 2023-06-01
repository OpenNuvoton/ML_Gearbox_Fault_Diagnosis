import numpy as np 
import pandas as pd
import os
import math

#class load_data:
def test():
    print('test')

### User decide functions for different data's format ###
def addCol_load(x, frame):
    frame['load'] = 10*x*np.ones((len(frame),1))
    return frame

def addCol_fault(anomaly, frame):
    frame['fault'] = anomaly*np.ones((len(frame),1))
    return frame

### Pull data from pth, and label with normal or anomaly ###
### Also have adding columns with customized function ###
def create_df_fromRaw(pth, anomaly, *funcs):
    
    file_list = []
    for file in os.listdir(pth):
        file_list.append(os.path.join(pth, file))
    file_list.sort()
    
    _DF = []
    _df = pd.DataFrame(_DF)
    for x in range(0, 10):
        frame = pd.read_csv(file_list[x])
        
        frame = funcs[0][0](x, frame)
        frame = funcs[0][1](anomaly, frame)
        
        _df = pd.concat([_df, frame], axis=0, ignore_index=True)
    
    return _df

### get data for MCU inference test latter, and this data will not be train.  
def create_test_train_df_fromRaw(pth, anomaly, test_len, win_size, *funcs):
    
    file_list = []
    for file in os.listdir(pth):
        file_list.append(os.path.join(pth, file))
    file_list.sort()
    
    _DF = []
    _df = pd.DataFrame(_DF)
    _df_test = pd.DataFrame(_DF)
    for x in range(0, 10):
        frame = pd.read_csv(file_list[x])
        
        frame = funcs[0][0](x, frame)
        frame = funcs[0][1](anomaly, frame)
        #print(frame.shape[0], math.floor(frame.shape[0]/300))
        
        _df_test = pd.concat([_df_test, frame[0:test_len*win_size]], axis=0, ignore_index=True)
        _df = pd.concat([_df, frame[test_len*win_size:]], axis=0, ignore_index=True)
        
    return _df, _df_test