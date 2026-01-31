"""
Author: Jii Kwon <jii.kwon125@gmail.com>
Seoul National University
Human Brain Function Laboratory 

Basic codes: Music imagery decoding
"""

#%% Import library
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import openpyxl as ec
import pickle
#%% Define function
class basic():
    def info_sub(self, SubNum, file_inform):
        info = sio.loadmat(file_inform)
        info = info['Info_Sub' + str(SubNum)]
        SubName = info['SubName'][0][0][0]
        NumCh = int(info['NumCh'])
        Bad_Ch = info['Bad_Ch'][0][0][0]
        Ch_list = info['Ch_list'][0][0][0]
        sub_fname = 'Sub'+str(SubNum)+'_'+SubName
        return sub_fname, SubName, NumCh, Ch_list, Bad_Ch
    
    
    def del_value(self, val_name,range_sd = np.arange(1,7)):
        for sd in range_sd:
            df_name = val_name+str(sd)
            del globals()[df_name]
            
    
    def export_excel(self, file_name, df, sheet_name='sheet1', idx=True):
        if os.path.isfile(file_name) == 0:
            wb = ec.Workbook()
            wb.save(file_name)
            df.to_excel(file_name, index=idx, sheet_name=sheet_name)
        else:
            writer = pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay')
            writer.workbook = ec.load_workbook(file_name)
            df.to_excel(writer, index=idx, sheet_name=sheet_name)
            writer.close()
    
    def load_pkl(self, fname):
        with open(fname, 'rb') as f:
            df = pickle.load(f)
    
        return df
    
    def set_folder(self, fold_path):
        fold_name = fold_path
        if not os.path.exists(fold_name): os.makedirs(fold_name)
        return fold_name
    
    def find_names(self, folder_name, pattern, type_ = 'dir'):
        import os
        import glob
        
        if type_ == 'dir':
            find_result = [f for f in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, f)) 
                      and f.startswith(pattern)]
        else: find_result = glob.glob(os.path.join(folder_name, pattern))
        
        return find_result
                
            
            
            
            