"""
Author: Jii Kwon <jii.kwon125@gmail.com>
Seoul National University
Human Brain Function Laboratory 

ECoG_Music : Epoching for single pitch classification
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


#%% Import Library
import numpy as np
import scipy.io as sio

from tqdm import trange
from scipy.signal import hilbert

from utils_basic_music import pd, sio, plt
from utils_basic_music import basic

from utils_etc_function import Utils 
from tqdm import trange
import pickle

utils = Utils()
basic = basic()
#%% Setting Analysis Parameter
standard = 'scale_degree' 
analysis = 'Imagery'
elec = 'Cortex'
margin_time = 1.0
rot = 0.5
BASE = 'norm_sd1'

Sub_Nums = np.arange(1,11)

bands = ['D', 'T', 'A', 'B', 'G', 'HG']

fold_select = os.path.join(folder_std, '01_select_electrode')
file_inform = r'E:\ECoG_Music\Subject_info.mat'

fold_feature = os.path.join(folder_std, '03_Extracted_Feature')
fold_feature_concat = os.path.join(fold_feature, 'concat_session_' + BASE)
if not os.path.exists(fold_feature_concat): os.makedirs(fold_feature_concat)

#%%Load Data
for sub_idx in trange(len(Sub_Nums)):
    SubNum = Sub_Nums[sub_idx]
    SubName, NumCh, Ch_list, Bad_Ch = basic.info_sub(SubNum,file_inform)
    sub_fname = 'Sub'+str(SubNum)+'_'+SubName
    

    fname = os.path.join(fold_select, 'case2_P1orP2', sub_fname+'.csv')
    selected = pd.read_csv(fname, index_col=0)
    
    selected = selected[selected['0']==True]
    name_list = selected['Name (aparc)']
    occ= name_list.str.contains("occi")
    
    selected2 = selected[occ==False]
    
    if SubNum == 4: Ch_list = Ch_list[:44]
    
    selected_Ch = Ch_list[selected2.index]
    
    selected = pd.read_csv(fname, index_col=0)
    selected_Ch2 = Ch_list[selected['0']]
    
    # Load Feature
    for bd in range(len(bands)):
        bname = str(bd+1)+'_'+bands[bd]
        fname = os.path.join(fold_feature,bname,BASE,sub_fname+'.mat')
        file = sio.loadmat(fname)
        Test_sess = int(file['Test_sess'][0])
        
        tmp_name = 'norm_C_sd'
        norm_C_sd1 = np.mean(file[tmp_name+str(1)], axis = 1)
        norm_C_sd2 = np.mean(file[tmp_name+str(2)], axis = 1)
        norm_C_sd3 = np.mean(file[tmp_name+str(3)], axis = 1)
        norm_C_sd4 = np.mean(file[tmp_name+str(4)], axis = 1)
        norm_C_sd5 = np.mean(file[tmp_name+str(5)], axis = 1)
        norm_C_sd6 = np.mean(file[tmp_name+str(6)], axis = 1)
        
        tmp_name = 'norm_D_sd'
        norm_D_sd1 = np.mean(file[tmp_name+str(1)], axis = 1)
        norm_D_sd2 = np.mean(file[tmp_name+str(2)], axis = 1)
        norm_D_sd3 = np.mean(file[tmp_name+str(3)], axis = 1)
        norm_D_sd4 = np.mean(file[tmp_name+str(4)], axis = 1)
        norm_D_sd5 = np.mean(file[tmp_name+str(5)], axis = 1)
        norm_D_sd6 = np.mean(file[tmp_name+str(6)], axis = 1)
        
        tmp_name = 'norm_B_sd'
        norm_B_sd1 = np.mean(file[tmp_name+str(1)], axis = 1)
        norm_B_sd2 = np.mean(file[tmp_name+str(2)], axis = 1)
        norm_B_sd3 = np.mean(file[tmp_name+str(3)], axis = 1)
        norm_B_sd4 = np.mean(file[tmp_name+str(4)], axis = 1)
        norm_B_sd5 = np.mean(file[tmp_name+str(5)], axis = 1)
        norm_B_sd6 = np.mean(file[tmp_name+str(6)], axis = 1)
        
        norm_C_sd1 = norm_C_sd1[selected2.index,:]
        norm_C_sd2 = norm_C_sd2[selected2.index,:]
        norm_C_sd3 = norm_C_sd3[selected2.index,:]
        norm_C_sd4 = norm_C_sd4[selected2.index,:]
        norm_C_sd5 = norm_C_sd5[selected2.index,:]
        norm_C_sd6 = norm_C_sd6[selected2.index,:]
        
        norm_D_sd1 = norm_D_sd1[selected2.index,:]
        norm_D_sd2 = norm_D_sd2[selected2.index,:]
        norm_D_sd3 = norm_D_sd3[selected2.index,:]
        norm_D_sd4 = norm_D_sd4[selected2.index,:]
        norm_D_sd5 = norm_D_sd5[selected2.index,:]
        norm_D_sd6 = norm_D_sd6[selected2.index,:]
        
        norm_B_sd1 = norm_B_sd1[selected2.index,:]
        norm_B_sd2 = norm_B_sd2[selected2.index,:]
        norm_B_sd3 = norm_B_sd3[selected2.index,:]
        norm_B_sd4 = norm_B_sd4[selected2.index,:]
        norm_B_sd5 = norm_B_sd5[selected2.index,:]
        norm_B_sd6 = norm_B_sd6[selected2.index,:]
        
        exec("bd_%s_C_sd1 = norm_C_sd1"%bands[bd])
        exec("bd_%s_C_sd2 = norm_C_sd2"%bands[bd])
        exec("bd_%s_C_sd3 = norm_C_sd3"%bands[bd])
        exec("bd_%s_C_sd4 = norm_C_sd4"%bands[bd])
        exec("bd_%s_C_sd5 = norm_C_sd5"%bands[bd])
        exec("bd_%s_C_sd6 = norm_C_sd6"%bands[bd])
        
        exec("bd_%s_D_sd1 = norm_D_sd1"%bands[bd])
        exec("bd_%s_D_sd2 = norm_D_sd2"%bands[bd])
        exec("bd_%s_D_sd3 = norm_D_sd3"%bands[bd])
        exec("bd_%s_D_sd4 = norm_D_sd4"%bands[bd])
        exec("bd_%s_D_sd5 = norm_D_sd5"%bands[bd])
        exec("bd_%s_D_sd6 = norm_D_sd6"%bands[bd])
        
        exec("bd_%s_B_sd1 = norm_B_sd1"%bands[bd])
        exec("bd_%s_B_sd2 = norm_B_sd2"%bands[bd])
        exec("bd_%s_B_sd3 = norm_B_sd3"%bands[bd])
        exec("bd_%s_B_sd4 = norm_B_sd4"%bands[bd])
        exec("bd_%s_B_sd5 = norm_B_sd5"%bands[bd])
        exec("bd_%s_B_sd6 = norm_B_sd6"%bands[bd])
        
    samples = [norm_B_sd1.shape[1], norm_B_sd2.shape[1], norm_B_sd3.shape[1], norm_B_sd4.shape[1], norm_B_sd5.shape[1], norm_B_sd6.shape[1]]
    samples = [int(x/int(Test_sess)) for x in samples]
    
    for sess in range(1, Test_sess+1):
        for bd in range(len(bands)):
            bname = str(bd)+'_'+bands[bd]
            tmp_sd1, tmp_sd2, tmp_sd3, tmp_sd4 = [], [], [], []
            tmp_sd5, tmp_sd6, tmp_sd7 = [], [], []
            
            exec("tmp_sd1 = bd_%s_C_sd1[:,samples[0]*(sess-1):samples[0]*(sess)]"%bands[bd])
            exec("tmp_sd1 = np.concatenate((tmp_sd1, bd_%s_D_sd1[:,samples[0]*(sess-1):samples[0]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd1 = np.concatenate((tmp_sd1, bd_%s_B_sd1[:,samples[0]*(sess-1):samples[0]*(sess)]), axis =1)"%bands[bd])
            
            exec("tmp_sd2 = bd_%s_C_sd2[:,samples[1]*(sess-1):samples[1]*(sess)]"%bands[bd])
            exec("tmp_sd2 = np.concatenate((tmp_sd2, bd_%s_D_sd2[:,samples[1]*(sess-1):samples[1]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd2 = np.concatenate((tmp_sd2, bd_%s_B_sd2[:,samples[1]*(sess-1):samples[1]*(sess)]), axis =1)"%bands[bd])
            
            
            exec("tmp_sd3 = bd_%s_C_sd3[:,samples[2]*(sess-1):samples[2]*(sess)]"%bands[bd])
            exec("tmp_sd3 = np.concatenate((tmp_sd3, bd_%s_D_sd3[:,samples[2]*(sess-1):samples[2]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd3 = np.concatenate((tmp_sd3, bd_%s_B_sd3[:,samples[2]*(sess-1):samples[2]*(sess)]), axis =1)"%bands[bd])
            
            exec("tmp_sd4 = bd_%s_C_sd4[:,samples[3]*(sess-1):samples[3]*(sess)]"%bands[bd])
            exec("tmp_sd4 = np.concatenate((tmp_sd4, bd_%s_D_sd4[:,samples[3]*(sess-1):samples[3]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd4 = np.concatenate((tmp_sd4, bd_%s_B_sd4[:,samples[3]*(sess-1):samples[3]*(sess)]), axis =1)"%bands[bd])
            
            exec("tmp_sd5 = bd_%s_C_sd5[:,samples[4]*(sess-1):samples[4]*(sess)]"%bands[bd])
            exec("tmp_sd5 = np.concatenate((tmp_sd5, bd_%s_D_sd5[:,samples[4]*(sess-1):samples[4]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd5 = np.concatenate((tmp_sd5, bd_%s_B_sd5[:,samples[4]*(sess-1):samples[4]*(sess)]), axis =1)"%bands[bd])
            
            exec("tmp_sd6 = bd_%s_C_sd6[:,samples[5]*(sess-1):samples[5]*(sess)]"%bands[bd])
            exec("tmp_sd6 = np.concatenate((tmp_sd6, bd_%s_D_sd6[:,samples[5]*(sess-1):samples[5]*(sess)]), axis =1)"%bands[bd])
            exec("tmp_sd6 = np.concatenate((tmp_sd6, bd_%s_B_sd6[:,samples[5]*(sess-1):samples[5]*(sess)]), axis =1)"%bands[bd])
            
            tmp_sd1 = pd.DataFrame(tmp_sd1).T
            tmp_sd2 = pd.DataFrame(tmp_sd2).T
            tmp_sd3 = pd.DataFrame(tmp_sd3).T
            tmp_sd4 = pd.DataFrame(tmp_sd4).T
            tmp_sd5 = pd.DataFrame(tmp_sd5).T
            tmp_sd6 = pd.DataFrame(tmp_sd6).T
            
            tmp_sd1['class'] ='SD1'
            tmp_sd2['class'] ='SD2'
            tmp_sd3['class'] ='SD3'
            tmp_sd4['class'] ='SD4'
            tmp_sd5['class'] ='SD5'
            tmp_sd6['class'] ='SD6'
            
            val_name ='tmp_sd'
            range_sd = np.arange(1,7)
            dataframes_dict = {
                   val_name + str(i): globals()[val_name + str(i)]  
                   for i in range_sd
                   }
            
            globals()['feature_sess_'+str(int(sess))+'_'+bands[bd]] = utils.create_df(dataframes_dict, val_name=val_name,range_sd = range_sd)
        
            feature_name = [bands[bd]+'_Ch'+str(int(k)) for k in selected_Ch]
            feature_name.extend(["class"])
            globals()['feature_sess_'+str(int(sess))+'_'+bands[bd]].columns = feature_name
    
    for sess in range(1, Test_sess+1):
        val_name =['feature_sess_'+str(int(sess))+'_']
        range_sd = ['D', 'T', 'A', 'B', 'G', 'HG']
        dataframes_dict = {
               val_name[0] + str(i): globals()[val_name[0] + str(i)]  
               for i in range_sd
               }
        
        concat_feature = utils.create_df(dataframes_dict, val_name=val_name[0], range_sd = range_sd,axis=1)
        concat_feature_class = concat_feature['class'].iloc[:,0]
        concat_feature.drop('class', axis =1, inplace=True)
        
        feautre_name = concat_feature.columns[:-1]
        class_list = list(dict.fromkeys(concat_feature_class)) 
        
        feature_dataset, class_names = concat_feature[feautre_name], concat_feature_class
        
        fname = os.path.join(fold_feature_concat, sub_fname+'_session_'+str(int(sess)) +'.pkl')
        with open(fname, 'wb') as f:
            pickle.dump({'train_data': feature_dataset, 'train_label': class_names, 'Test_sess': Test_sess}, f)