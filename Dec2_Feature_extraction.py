"""
Author: Jii Kwon <jii.kwon125@gmail.com>
Seoul National University
Human Brain Function Laboratory 

ECoG_Music : Epoching for single pitch classification
"""


#%% Go to root
import os.path
import os
folder_std = r'E:\ECoG_Music\2304_Music_Imagery_Decoding'
os.chdir(folder_std)
os.getcwd()

#%% Import Library
import numpy as np
import scipy.io as sio

from tqdm import trange
from scipy.signal import hilbert

#%% Setting Analysis Parameter
standard = 'scale_degree' 
analysis = 'Imagery'
elec = 'Cortex'
margin_time = 1.0
rot = 0.5

Sub_Nums = np.arange(1,11)

bands = ['D', 'T', 'A', 'B', 'G', 'HG']

file_inform = r'E:\ECoG_Music\Subject_info.mat'

fold_select = os.path.join(folder_std, '01_select_electrode')
fold_epoch = os.path.join(folder_std, '02_Epoch_Classification')
fold_feature = os.path.join(folder_std, '03_Extracted_Feature')
if not os.path.exists(fold_feature): os.makedirs(fold_feature)

#%% Define fuction
def info_sub(SubNum):
    info = sio.loadmat(file_inform)
    info = info['Info_Sub' + str(SubNum)]
    SubName = info['SubName'][0][0][0]
    NumCh = int(info['NumCh'])
    Bad_Ch = info['Bad_Ch'][0][0][0]
    Ch_list = info['Ch_list'][0][0][0]
    
    return SubName, NumCh, Ch_list, Bad_Ch


def hilb_3d(data, rot=0.5, margin_time=0, srate=2000):
    x1 = data.shape[0]
    x2 = data.shape[2]
    hilb_data1 = np.zeros([x1, data.shape[1], x2])
    for a in range(x1):
        for b in range(x2):
            hilb_data1[a,:,b] = abs(hilbert(data[a,:,b]))
    
    hilb_data = hilb_data1[:,int(margin_time*srate):int((margin_time+rot)*srate),:]
    return hilb_data

#%%Load Data

for bd in trange(len(bands)):
    spec_bd = str(bd+1)+'_'+bands[bd]
    
    fold_epoch_bd = os.path.join(fold_epoch, spec_bd)
    
    fold_feature_bd = os.path.join(fold_feature, spec_bd)
    if not os.path.exists(fold_feature_bd): os.makedirs(fold_feature_bd)
    
    for sub_idx in trange(len(Sub_Nums)):
        SubNum = Sub_Nums[sub_idx]
        
        SubName, NumCh, Ch_list, Bad_Ch = info_sub(SubNum)
       
        fname = os.path.join(fold_epoch_bd, 'Sub'+str(SubNum)+'_'+SubName+'.mat')
        
        
        epoch = sio.loadmat(fname)
        
        epoch_C_sd1 = epoch['epoch_C_sd1']
        epoch_C_sd2 = epoch['epoch_C_sd2']
        epoch_C_sd3 = epoch['epoch_C_sd3']
        epoch_C_sd4 = epoch['epoch_C_sd4']
        epoch_C_sd5 = epoch['epoch_C_sd5']
        epoch_C_sd6 = epoch['epoch_C_sd6']
        epoch_C_sd7 = epoch['epoch_C_sd7']
        epoch_C_sd8 = epoch['epoch_C_sd8']
        
        epoch_D_sd1 = epoch['epoch_D_sd1']
        epoch_D_sd2 = epoch['epoch_D_sd2']
        epoch_D_sd3 = epoch['epoch_D_sd3']
        epoch_D_sd4 = epoch['epoch_D_sd4']
        epoch_D_sd5 = epoch['epoch_D_sd5']
        epoch_D_sd6 = epoch['epoch_D_sd6']
        epoch_D_sd7 = epoch['epoch_D_sd7']
        epoch_D_sd8 = epoch['epoch_D_sd8']
        
        epoch_B_sd1 = epoch['epoch_B_sd1']
        epoch_B_sd2 = epoch['epoch_B_sd2']
        epoch_B_sd3 = epoch['epoch_B_sd3']
        epoch_B_sd4 = epoch['epoch_B_sd4']
        epoch_B_sd5 = epoch['epoch_B_sd5']
        epoch_B_sd6 = epoch['epoch_B_sd6']
        epoch_B_sd7 = epoch['epoch_B_sd7']
        epoch_B_sd8 = epoch['epoch_B_sd8']
        
        margin_time = epoch['margin_time']
        Bad_Ch = epoch['Bad_Ch']
        srate = epoch['srate']
        
        del epoch
        
        hilb_C_sd1 = hilb_3d(epoch_C_sd1, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd2 = hilb_3d(epoch_C_sd2, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd3 = hilb_3d(epoch_C_sd3, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd4 = hilb_3d(epoch_C_sd4, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd5 = hilb_3d(epoch_C_sd5, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd6 = hilb_3d(epoch_C_sd6, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd7 = hilb_3d(epoch_C_sd7, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_C_sd8 = hilb_3d(epoch_C_sd8, rot=rot, margin_time=int(margin_time), srate=int(srate))
        
        hilb_D_sd1 = hilb_3d(epoch_D_sd1, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd2 = hilb_3d(epoch_D_sd2, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd3 = hilb_3d(epoch_D_sd3, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd4 = hilb_3d(epoch_D_sd4, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd5 = hilb_3d(epoch_D_sd5, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd6 = hilb_3d(epoch_D_sd6, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd7 = hilb_3d(epoch_D_sd7, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_D_sd8 = hilb_3d(epoch_D_sd8, rot=rot, margin_time=int(margin_time), srate=int(srate))
        
        hilb_B_sd1 = hilb_3d(epoch_B_sd1, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd2 = hilb_3d(epoch_B_sd2, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd3 = hilb_3d(epoch_B_sd3, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd4 = hilb_3d(epoch_B_sd4, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd5 = hilb_3d(epoch_B_sd5, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd6 = hilb_3d(epoch_B_sd6, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd7 = hilb_3d(epoch_B_sd7, rot=rot, margin_time=int(margin_time), srate=int(srate))
        hilb_B_sd8 = hilb_3d(epoch_B_sd8, rot=rot, margin_time=int(margin_time), srate=int(srate))
        
        del epoch_C_sd1, epoch_C_sd2, epoch_C_sd3, epoch_C_sd4, epoch_C_sd5, epoch_C_sd6, epoch_C_sd7, epoch_C_sd8
        del epoch_D_sd1, epoch_D_sd2, epoch_D_sd3, epoch_D_sd4, epoch_D_sd5, epoch_D_sd6, epoch_D_sd7, epoch_D_sd8
        del epoch_B_sd1, epoch_B_sd2, epoch_B_sd3, epoch_B_sd4, epoch_B_sd5, epoch_B_sd6, epoch_B_sd7, epoch_B_sd8
        
        fold_feature_bd1 = os.path.join(fold_feature_bd, 'Org_hilb')
        if not os.path.exists(fold_feature_bd1): os.makedirs(fold_feature_bd1)
        
        fname = os.path.join(fold_feature_bd1, 'Sub'+str(SubNum)+'_'+SubName+'.mat')
        sio.savemat(fname,{
                    'hilb_C_sd1': hilb_C_sd1, 'hilb_C_sd2': hilb_C_sd2, 'hilb_C_sd3': hilb_C_sd3, 
                    'hilb_C_sd4': hilb_C_sd4, 'hilb_C_sd5': hilb_C_sd5, 'hilb_C_sd6': hilb_C_sd6, 
                    'hilb_C_sd7': hilb_C_sd7, 'hilb_C_sd8': hilb_C_sd8,
                    'hilb_D_sd1': hilb_D_sd1, 'hilb_D_sd2': hilb_D_sd2, 'hilb_D_sd3': hilb_D_sd3, 
                    'hilb_D_sd4': hilb_D_sd4, 'hilb_D_sd5': hilb_D_sd5, 'hilb_D_sd6': hilb_D_sd6, 
                    'hilb_D_sd7': hilb_D_sd7, 'hilb_D_sd8': hilb_D_sd8,
                    'hilb_B_sd1': hilb_B_sd1, 'hilb_B_sd2': hilb_B_sd2, 'hilb_B_sd3': hilb_B_sd3, 
                    'hilb_B_sd4': hilb_B_sd4, 'hilb_B_sd5': hilb_B_sd5, 'hilb_B_sd6': hilb_B_sd6, 
                    'hilb_B_sd7': hilb_B_sd7, 'hilb_B_sd8': hilb_B_sd8,
                    })
        
        # normalization : SD1
        norm_C_base = np.mean(hilb_C_sd1, axis=1)
        norm_C_base = np.mean(norm_C_base, axis=1)
        
        norm_C_sd1 = np.zeros((hilb_C_sd1.shape[0], hilb_C_sd1.shape[1], hilb_C_sd1.shape[2]))
        norm_C_sd2 = np.zeros((hilb_C_sd2.shape[0], hilb_C_sd2.shape[1], hilb_C_sd2.shape[2]))
        norm_C_sd3 = np.zeros((hilb_C_sd3.shape[0], hilb_C_sd3.shape[1], hilb_C_sd3.shape[2]))
        norm_C_sd4 = np.zeros((hilb_C_sd4.shape[0], hilb_C_sd4.shape[1], hilb_C_sd4.shape[2]))
        norm_C_sd5 = np.zeros((hilb_C_sd5.shape[0], hilb_C_sd5.shape[1], hilb_C_sd5.shape[2]))
        norm_C_sd6 = np.zeros((hilb_C_sd6.shape[0], hilb_C_sd6.shape[1], hilb_C_sd6.shape[2]))
        norm_C_sd7 = np.zeros((hilb_C_sd7.shape[0], hilb_C_sd7.shape[1], hilb_C_sd7.shape[2]))
        norm_C_sd8 = np.zeros((hilb_C_sd8.shape[0], hilb_C_sd8.shape[1], hilb_C_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_C_sd1.shape[2]):
                norm_C_sd1[ch,:,tt] = hilb_C_sd1[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd2.shape[2]):
                norm_C_sd2[ch,:,tt] = hilb_C_sd2[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd3.shape[2]):
                norm_C_sd3[ch,:,tt] = hilb_C_sd3[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd4.shape[2]):
                norm_C_sd4[ch,:,tt] = hilb_C_sd4[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd5.shape[2]):
                norm_C_sd5[ch,:,tt] = hilb_C_sd5[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd6.shape[2]):
                norm_C_sd6[ch,:,tt] = hilb_C_sd6[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd7.shape[2]):
                norm_C_sd7[ch,:,tt] = hilb_C_sd7[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd8.shape[2]):
                norm_C_sd8[ch,:,tt] = hilb_C_sd8[ch,:,tt] - norm_C_base[ch]
                
        norm_D_base = np.mean(hilb_D_sd1, axis=1)
        norm_D_base = np.mean(norm_D_base, axis=1)
        
        norm_D_sd1 = np.zeros((hilb_D_sd1.shape[0], hilb_D_sd1.shape[1], hilb_D_sd1.shape[2]))
        norm_D_sd2 = np.zeros((hilb_D_sd2.shape[0], hilb_D_sd2.shape[1], hilb_D_sd2.shape[2]))
        norm_D_sd3 = np.zeros((hilb_D_sd3.shape[0], hilb_D_sd3.shape[1], hilb_D_sd3.shape[2]))
        norm_D_sd4 = np.zeros((hilb_D_sd4.shape[0], hilb_D_sd4.shape[1], hilb_D_sd4.shape[2]))
        norm_D_sd5 = np.zeros((hilb_D_sd5.shape[0], hilb_D_sd5.shape[1], hilb_D_sd5.shape[2]))
        norm_D_sd6 = np.zeros((hilb_D_sd6.shape[0], hilb_D_sd6.shape[1], hilb_D_sd6.shape[2]))
        norm_D_sd7 = np.zeros((hilb_D_sd7.shape[0], hilb_D_sd7.shape[1], hilb_D_sd7.shape[2]))
        norm_D_sd8 = np.zeros((hilb_D_sd8.shape[0], hilb_D_sd8.shape[1], hilb_D_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_D_sd1.shape[2]):
                norm_D_sd1[ch,:,tt] = hilb_D_sd1[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd2.shape[2]):
                norm_D_sd2[ch,:,tt] = hilb_D_sd2[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd3.shape[2]):
                norm_D_sd3[ch,:,tt] = hilb_D_sd3[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd4.shape[2]):
                norm_D_sd4[ch,:,tt] = hilb_D_sd4[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd5.shape[2]):
                norm_D_sd5[ch,:,tt] = hilb_D_sd5[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd6.shape[2]):
                norm_D_sd6[ch,:,tt] = hilb_D_sd6[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd7.shape[2]):
                norm_D_sd7[ch,:,tt] = hilb_D_sd7[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd8.shape[2]):
                norm_D_sd8[ch,:,tt] = hilb_D_sd8[ch,:,tt] - norm_D_base[ch]
        
        norm_B_base = np.mean(hilb_B_sd1, axis=1)
        norm_B_base = np.mean(norm_B_base, axis=1)
        
        norm_B_sd1 = np.zeros((hilb_B_sd1.shape[0], hilb_B_sd1.shape[1], hilb_B_sd1.shape[2]))
        norm_B_sd2 = np.zeros((hilb_B_sd2.shape[0], hilb_B_sd2.shape[1], hilb_B_sd2.shape[2]))
        norm_B_sd3 = np.zeros((hilb_B_sd3.shape[0], hilb_B_sd3.shape[1], hilb_B_sd3.shape[2]))
        norm_B_sd4 = np.zeros((hilb_B_sd4.shape[0], hilb_B_sd4.shape[1], hilb_B_sd4.shape[2]))
        norm_B_sd5 = np.zeros((hilb_B_sd5.shape[0], hilb_B_sd5.shape[1], hilb_B_sd5.shape[2]))
        norm_B_sd6 = np.zeros((hilb_B_sd6.shape[0], hilb_B_sd6.shape[1], hilb_B_sd6.shape[2]))
        norm_B_sd7 = np.zeros((hilb_B_sd7.shape[0], hilb_B_sd7.shape[1], hilb_B_sd7.shape[2]))
        norm_B_sd8 = np.zeros((hilb_B_sd8.shape[0], hilb_B_sd8.shape[1], hilb_B_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_B_sd1.shape[2]):
                norm_B_sd1[ch,:,tt] = hilb_B_sd1[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd2.shape[2]):
                norm_B_sd2[ch,:,tt] = hilb_B_sd2[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd3.shape[2]):
                norm_B_sd3[ch,:,tt] = hilb_B_sd3[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd4.shape[2]):
                norm_B_sd4[ch,:,tt] = hilb_B_sd4[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd5.shape[2]):
                norm_B_sd5[ch,:,tt] = hilb_B_sd5[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd6.shape[2]):
                norm_B_sd6[ch,:,tt] = hilb_B_sd6[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd7.shape[2]):
                norm_B_sd7[ch,:,tt] = hilb_B_sd7[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd8.shape[2]):
                norm_B_sd8[ch,:,tt] = hilb_B_sd8[ch,:,tt] - norm_B_base[ch]
        
        fold_feature_bd2 = os.path.join(fold_feature_bd, 'norm_sd1')
        if not os.path.exists(fold_feature_bd2): os.makedirs(fold_feature_bd2)
        
        fname = os.path.join(fold_feature_bd2, 'Sub'+str(SubNum)+'_'+SubName+'.mat')
        sio.savemat(fname,{ 'norm':'sd1',
                    'norm_C_sd1': norm_C_sd1, 'norm_C_sd2': norm_C_sd2, 'norm_C_sd3': norm_C_sd3, 
                    'norm_C_sd4': norm_C_sd4, 'norm_C_sd5': norm_C_sd5, 'norm_C_sd6': norm_C_sd6, 
                    'norm_C_sd7': norm_C_sd7, 'norm_C_sd8': norm_C_sd8,
                    'norm_D_sd1': norm_D_sd1, 'norm_D_sd2': norm_D_sd2, 'norm_D_sd3': norm_D_sd3, 
                    'norm_D_sd4': norm_D_sd4, 'norm_D_sd5': norm_D_sd5, 'norm_D_sd6': norm_D_sd6, 
                    'norm_D_sd7': norm_D_sd7, 'norm_D_sd8': norm_D_sd8,
                    'norm_B_sd1': norm_B_sd1, 'norm_B_sd2': norm_B_sd2, 'norm_B_sd3': norm_B_sd3, 
                    'norm_B_sd4': norm_B_sd4, 'norm_B_sd5': norm_B_sd5, 'norm_B_sd6': norm_B_sd6, 
                    'norm_B_sd7': norm_B_sd7, 'norm_B_sd8': norm_B_sd8,
                    })                                              
        
        del norm_C_base, norm_D_base, norm_B_base
        del norm_C_sd1, norm_C_sd2, norm_C_sd3, norm_C_sd4, norm_C_sd5, norm_C_sd6, norm_C_sd7, norm_C_sd8
        del norm_D_sd1, norm_D_sd2, norm_D_sd3, norm_D_sd4, norm_D_sd5, norm_D_sd6, norm_D_sd7, norm_D_sd8
        del norm_B_sd1, norm_B_sd2, norm_B_sd3, norm_B_sd4, norm_B_sd5, norm_B_sd6, norm_B_sd7, norm_B_sd8
        
        
        # normalization : SD7
        norm_C_base = np.mean(hilb_C_sd7, axis=1)
        norm_C_base = np.mean(norm_C_base, axis=1)
        
        norm_C_sd1 = np.zeros((hilb_C_sd1.shape[0], hilb_C_sd1.shape[1], hilb_C_sd1.shape[2]))
        norm_C_sd2 = np.zeros((hilb_C_sd2.shape[0], hilb_C_sd2.shape[1], hilb_C_sd2.shape[2]))
        norm_C_sd3 = np.zeros((hilb_C_sd3.shape[0], hilb_C_sd3.shape[1], hilb_C_sd3.shape[2]))
        norm_C_sd4 = np.zeros((hilb_C_sd4.shape[0], hilb_C_sd4.shape[1], hilb_C_sd4.shape[2]))
        norm_C_sd5 = np.zeros((hilb_C_sd5.shape[0], hilb_C_sd5.shape[1], hilb_C_sd5.shape[2]))
        norm_C_sd6 = np.zeros((hilb_C_sd6.shape[0], hilb_C_sd6.shape[1], hilb_C_sd6.shape[2]))
        norm_C_sd7 = np.zeros((hilb_C_sd7.shape[0], hilb_C_sd7.shape[1], hilb_C_sd7.shape[2]))
        norm_C_sd8 = np.zeros((hilb_C_sd8.shape[0], hilb_C_sd8.shape[1], hilb_C_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_C_sd1.shape[2]):
                norm_C_sd1[ch,:,tt] = hilb_C_sd1[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd2.shape[2]):
                norm_C_sd2[ch,:,tt] = hilb_C_sd2[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd3.shape[2]):
                norm_C_sd3[ch,:,tt] = hilb_C_sd3[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd4.shape[2]):
                norm_C_sd4[ch,:,tt] = hilb_C_sd4[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd5.shape[2]):
                norm_C_sd5[ch,:,tt] = hilb_C_sd5[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd6.shape[2]):
                norm_C_sd6[ch,:,tt] = hilb_C_sd6[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd7.shape[2]):
                norm_C_sd7[ch,:,tt] = hilb_C_sd7[ch,:,tt] - norm_C_base[ch]
            for tt in range(hilb_C_sd8.shape[2]):
                norm_C_sd8[ch,:,tt] = hilb_C_sd8[ch,:,tt] - norm_C_base[ch]
                
        norm_D_base = np.mean(hilb_D_sd7, axis=1)
        norm_D_base = np.mean(norm_D_base, axis=1)
        
        norm_D_sd1 = np.zeros((hilb_D_sd1.shape[0], hilb_D_sd1.shape[1], hilb_D_sd1.shape[2]))
        norm_D_sd2 = np.zeros((hilb_D_sd2.shape[0], hilb_D_sd2.shape[1], hilb_D_sd2.shape[2]))
        norm_D_sd3 = np.zeros((hilb_D_sd3.shape[0], hilb_D_sd3.shape[1], hilb_D_sd3.shape[2]))
        norm_D_sd4 = np.zeros((hilb_D_sd4.shape[0], hilb_D_sd4.shape[1], hilb_D_sd4.shape[2]))
        norm_D_sd5 = np.zeros((hilb_D_sd5.shape[0], hilb_D_sd5.shape[1], hilb_D_sd5.shape[2]))
        norm_D_sd6 = np.zeros((hilb_D_sd6.shape[0], hilb_D_sd6.shape[1], hilb_D_sd6.shape[2]))
        norm_D_sd7 = np.zeros((hilb_D_sd7.shape[0], hilb_D_sd7.shape[1], hilb_D_sd7.shape[2]))
        norm_D_sd8 = np.zeros((hilb_D_sd8.shape[0], hilb_D_sd8.shape[1], hilb_D_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_D_sd1.shape[2]):
                norm_D_sd1[ch,:,tt] = hilb_D_sd1[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd2.shape[2]):
                norm_D_sd2[ch,:,tt] = hilb_D_sd2[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd3.shape[2]):
                norm_D_sd3[ch,:,tt] = hilb_D_sd3[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd4.shape[2]):
                norm_D_sd4[ch,:,tt] = hilb_D_sd4[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd5.shape[2]):
                norm_D_sd5[ch,:,tt] = hilb_D_sd5[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd6.shape[2]):
                norm_D_sd6[ch,:,tt] = hilb_D_sd6[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd7.shape[2]):
                norm_D_sd7[ch,:,tt] = hilb_D_sd7[ch,:,tt] - norm_D_base[ch]
            for tt in range(hilb_D_sd8.shape[2]):
                norm_D_sd8[ch,:,tt] = hilb_D_sd8[ch,:,tt] - norm_D_base[ch]
        
        norm_B_base = np.mean(hilb_B_sd7, axis=1)
        norm_B_base = np.mean(norm_B_base, axis=1)
        
        norm_B_sd1 = np.zeros((hilb_B_sd1.shape[0], hilb_B_sd1.shape[1], hilb_B_sd1.shape[2]))
        norm_B_sd2 = np.zeros((hilb_B_sd2.shape[0], hilb_B_sd2.shape[1], hilb_B_sd2.shape[2]))
        norm_B_sd3 = np.zeros((hilb_B_sd3.shape[0], hilb_B_sd3.shape[1], hilb_B_sd3.shape[2]))
        norm_B_sd4 = np.zeros((hilb_B_sd4.shape[0], hilb_B_sd4.shape[1], hilb_B_sd4.shape[2]))
        norm_B_sd5 = np.zeros((hilb_B_sd5.shape[0], hilb_B_sd5.shape[1], hilb_B_sd5.shape[2]))
        norm_B_sd6 = np.zeros((hilb_B_sd6.shape[0], hilb_B_sd6.shape[1], hilb_B_sd6.shape[2]))
        norm_B_sd7 = np.zeros((hilb_B_sd7.shape[0], hilb_B_sd7.shape[1], hilb_B_sd7.shape[2]))
        norm_B_sd8 = np.zeros((hilb_B_sd8.shape[0], hilb_B_sd8.shape[1], hilb_B_sd8.shape[2]))
        
        for ch in range(len(Ch_list)):
            for tt in range(hilb_B_sd1.shape[2]):
                norm_B_sd1[ch,:,tt] = hilb_B_sd1[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd2.shape[2]):
                norm_B_sd2[ch,:,tt] = hilb_B_sd2[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd3.shape[2]):
                norm_B_sd3[ch,:,tt] = hilb_B_sd3[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd4.shape[2]):
                norm_B_sd4[ch,:,tt] = hilb_B_sd4[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd5.shape[2]):
                norm_B_sd5[ch,:,tt] = hilb_B_sd5[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd6.shape[2]):
                norm_B_sd6[ch,:,tt] = hilb_B_sd6[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd7.shape[2]):
                norm_B_sd7[ch,:,tt] = hilb_B_sd7[ch,:,tt] - norm_B_base[ch]
            for tt in range(hilb_B_sd8.shape[2]):
                norm_B_sd8[ch,:,tt] = hilb_B_sd8[ch,:,tt] - norm_B_base[ch]
        
        fold_feature_bd3 = os.path.join(fold_feature_bd, 'norm_sd7')
        if not os.path.exists(fold_feature_bd3): os.makedirs(fold_feature_bd3)
        
        fname = os.path.join(fold_feature_bd3, 'Sub'+str(SubNum)+'_'+SubName+'.mat')
        sio.savemat(fname,{ 'norm':'sd7',
                    'norm_C_sd1': norm_C_sd1, 'norm_C_sd2': norm_C_sd2, 'norm_C_sd3': norm_C_sd3, 
                    'norm_C_sd4': norm_C_sd4, 'norm_C_sd5': norm_C_sd5, 'norm_C_sd6': norm_C_sd6, 
                    'norm_C_sd7': norm_C_sd7, 'norm_C_sd8': norm_C_sd8,
                    'norm_D_sd1': norm_D_sd1, 'norm_D_sd2': norm_D_sd2, 'norm_D_sd3': norm_D_sd3, 
                    'norm_D_sd4': norm_D_sd4, 'norm_D_sd5': norm_D_sd5, 'norm_D_sd6': norm_D_sd6, 
                    'norm_D_sd7': norm_D_sd7, 'norm_D_sd8': norm_D_sd8,
                    'norm_B_sd1': norm_B_sd1, 'norm_B_sd2': norm_B_sd2, 'norm_B_sd3': norm_B_sd3, 
                    'norm_B_sd4': norm_B_sd4, 'norm_B_sd5': norm_B_sd5, 'norm_B_sd6': norm_B_sd6, 
                    'norm_B_sd7': norm_B_sd7, 'norm_B_sd8': norm_B_sd8,
                    })        
        
        del norm_C_base, norm_D_base, norm_B_base
        del norm_C_sd1, norm_C_sd2, norm_C_sd3, norm_C_sd4, norm_C_sd5, norm_C_sd6, norm_C_sd7, norm_C_sd8
        del norm_D_sd1, norm_D_sd2, norm_D_sd3, norm_D_sd4, norm_D_sd5, norm_D_sd6, norm_D_sd7, norm_D_sd8
        del norm_B_sd1, norm_B_sd2, norm_B_sd3, norm_B_sd4, norm_B_sd5, norm_B_sd6, norm_B_sd7, norm_B_sd8
        
        del hilb_C_sd1, hilb_C_sd2, hilb_C_sd3, hilb_C_sd4, hilb_C_sd5, hilb_C_sd6, hilb_C_sd7, hilb_C_sd8
        del hilb_D_sd1, hilb_D_sd2, hilb_D_sd3, hilb_D_sd4, hilb_D_sd5, hilb_D_sd6, hilb_D_sd7, hilb_D_sd8
        del hilb_B_sd1, hilb_B_sd2, hilb_B_sd3, hilb_B_sd4, hilb_B_sd5, hilb_B_sd6, hilb_B_sd7, hilb_B_sd8
        