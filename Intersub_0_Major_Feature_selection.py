"""
Author: Jii Kwon <jii.kwon125@gmail.com>
Seoul National University
Human Brain Function Laboratory 

ECoG_Music : Intersubject model
"""
#%% Go to root
import os.path
import os

serv = input("Window:0 Linux:1 -")

if serv == '0':
    serv_name = 'E:\\'
else:
    serv_name = '/mnt/e'
    
folder_std = os.path.join(serv_name, 'ECoG_Music', '2304_Music_Imagery_Decoding')
folder_scripts = os.path.join(folder_std, '00_scripts')
    
os.chdir(folder_scripts)
os.getcwd()

#%% import library
from utils_basic_music import np, plt, sio, pd
from utils_basic_music import basic
from utils_etc_function import Utils 
from utils_ml_model import ml_models
from tqdm import trange
from collections import Counter
import pickle
import glob

utils = Utils()
ml_models = ml_models()
basic = basic()

#%% Setting Analysis Parameter [adjusting param]
# Decoding param
BASE = 'norm_sd1'
mean_max = 'mean'
kfold = 7

random_state = 1209

aug_val = 1.5 # None or number
param_set = None #Boolean : getting the gridsearch result
selecting_method  = 'RF'

#%% Setting Analysis Basic Parameter [not necessaries of adjusting param]
# Basic Param
Sub_Nums = np.arange(1,11)
bands = ['D', 'T', 'A', 'B', 'G', 'HG']
band_frqs = [(1,4), (4,7), (8,12), (12,30), (30,59), (61,150)]

file_inform = os.path.join(serv_name, 'ECoG_Music', 'Subject_info.mat')

# Set folder name
fold_feature_concat = os.path.join(folder_std, '03_Extracted_Feature','concat_session_' + BASE)

if aug_val == None: fold_class_FW = os.path.join(folder_std, '04_Clf_FW','sess1-8['+'aug_None_FI]_non_scaling')
else: fold_class_FW = os.path.join(folder_std, '04_Clf_FW','sess1-8['+'aug_'+str(aug_val)+'_FI]_non_scaling')