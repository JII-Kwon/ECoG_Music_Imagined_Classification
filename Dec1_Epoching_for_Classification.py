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
import pandas as pd
import scipy.io as sio
import mat73
import glob

from tqdm import trange

#%% Setting Analysis Parameter
analysis = 'Imagery'
standard = 'scale_degree'
margin_time = 1.0
rot = 0.5

Sub_Nums = np.arange(1,11)

bands = ['D', 'T', 'A', 'B', 'G', 'HG']

file_inform = r'E:\01_ECoG\Tot_subject_Music.mat'
fold_mat = r'E:\ECoG_Music\01_mat_file'

file_inform = r'E:\ECoG_Music\Subject_info.mat'

fold_select = os.path.join(folder_std, '01_select_electrode')
fold_epoch = os.path.join(folder_std, '02_Epoch_Classification')
if not os.path.exists(fold_epoch): os.makedirs(fold_epoch)

#%% Define fuction
def info_sub(SubNum):
    info = sio.loadmat(file_inform)
    info = info['Info_Sub' + str(SubNum)]
    SubName = info['SubName'][0][0][0]
    NumCh = int(info['NumCh'])
    Bad_Ch = info['Bad_Ch'][0][0][0]
    Ch_list = info['Ch_list'][0][0][0]
    
    return SubName, NumCh, Ch_list, Bad_Ch

#%%Load Data
for bd in trange(len(bands)):
    spec_bd = str(bd+1)+'_'+bands[bd]
    fold_mat_bd = os.path.join(fold_mat, spec_bd)
    
    fold_epoch_bd = os.path.join(fold_epoch, spec_bd)
    if not os.path.exists(fold_epoch_bd): os.makedirs(fold_epoch_bd)
    
    for sub_idx in trange(len(Sub_Nums)):
        SubNum = Sub_Nums[sub_idx]
        
        SubName, NumCh, Ch_list, Bad_Ch = info_sub(SubNum)
        SubToken = 'Sub' + str(SubNum) + '_' + SubName
        
        fn_mat = glob.glob(os.path.join(fold_mat_bd, spec_bd + '_' + SubToken+'_sess*.mat'))
        
        epoch_C_sd1 = []
        epoch_C_sd2 = []
        epoch_C_sd3 = []
        epoch_C_sd4 = []
        epoch_C_sd5 = []
        epoch_C_sd6 = []
        epoch_C_sd7 = []
        epoch_C_sd8 = []
        
        epoch_D_sd1 = []
        epoch_D_sd2 = []
        epoch_D_sd3 = []
        epoch_D_sd4 = []
        epoch_D_sd5 = []
        epoch_D_sd6 = []
        epoch_D_sd7 = []
        epoch_D_sd8 = []
        
        epoch_B_sd1 = []
        epoch_B_sd2 = []
        epoch_B_sd3 = []
        epoch_B_sd4 = []
        epoch_B_sd5 = []
        epoch_B_sd6 = []
        epoch_B_sd7 = []
        epoch_B_sd8 = []
        
        for ss, fn in enumerate(fn_mat):
            sess = ss +1
            
            filename = os.path.join(fn)
            #if bd == 6 : filename = os.path.join(fold_mat_bd, 'Cortex_1_'+bands[bd]+'_Sub'+str(SubNum)+'_'+SubName+'_sess'+str(sess)+'.mat')
            filename = mat73.loadmat(filename)
            
            Bad_Ch = filename['Bad_Ch']
            data = filename['fdata']
            trig = filename['trig']
            srate = filename['srate']
            
            del filename
            
            trig = pd.DataFrame.from_dict(trig)
            trig.drop(['accept', 'accept_ev1', 'latency'], axis = 1 , inplace = True)
            trig.rename(columns={'keyboard':'major', 'code':'song', 'epochevent':'scale_degree',
                                 'type':'absolute_pitch','keypad_accept':'proceeding','accuracy':'pitch_chroma'}, inplace=True)
            trig_header = trig.columns.tolist()     
            trig = trig.apply(pd.Series.explode)
            
            trig = pd.DataFrame.to_numpy(trig)
        
            od_stim = trig_header.index('stimtype') #1: Music listening 2: Imagery 3: Production
            od_task = trig_header.index(standard)
            od_maj = trig_header.index('major')
            od_offset = trig_header.index('offset')
            
            event_C_sd1 = []
            event_C_sd2 = []
            event_C_sd3 = []
            event_C_sd4 = []
            event_C_sd5 = []
            event_C_sd6 = []
            event_C_sd7 = []
            event_C_sd8 = []
            
            event_D_sd1 = []
            event_D_sd2 = []
            event_D_sd3 = []
            event_D_sd4 = []
            event_D_sd5 = []
            event_D_sd6 = []
            event_D_sd7 = []
            event_D_sd8 = []
            
            event_B_sd1 = []
            event_B_sd2 = []
            event_B_sd3 = []
            event_B_sd4 = []
            event_B_sd5 = []
            event_B_sd6 = []
            event_B_sd7 = []
            event_B_sd8 = []
            
            for tt in range(len(trig)-1):
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 1: event_C_sd1.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 2: event_C_sd2.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 3: event_C_sd3.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 4: event_C_sd4.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 5: event_C_sd5.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 6: event_C_sd6.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 7: event_C_sd7.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 1 and trig[tt,od_task] == 8: event_C_sd8.append(tt)
                
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 1: event_D_sd1.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 2: event_D_sd2.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 3: event_D_sd3.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 4: event_D_sd4.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 5: event_D_sd5.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 6: event_D_sd6.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 7: event_D_sd7.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 2 and trig[tt,od_task] == 8: event_D_sd8.append(tt)
                
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 1: event_B_sd1.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 2: event_B_sd2.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 3: event_B_sd3.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 4: event_B_sd4.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 5: event_B_sd5.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 6: event_B_sd6.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 7: event_B_sd7.append(tt)
                if trig[tt,od_stim] == 2 and trig[tt, od_maj] == 3 and trig[tt,od_task] == 8: event_B_sd8.append(tt)
            
            tmp_C_sd1 = np.array([])
            for tt in range(len(event_C_sd1)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd1[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd1 = ttmp
                elif tt == 1 : tmp_C_sd1 = np.concatenate((tmp_C_sd1[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd1 = np.concatenate((tmp_C_sd1, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_C_sd2 = np.array([])
            for tt in range(len(event_C_sd2)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd2[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd2 = ttmp
                elif tt == 1 : tmp_C_sd2 = np.concatenate((tmp_C_sd2[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd2 = np.concatenate((tmp_C_sd2, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_C_sd3 = np.array([])
            for tt in range(len(event_C_sd3)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd3[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd3 = ttmp
                elif tt == 1 : tmp_C_sd3 = np.concatenate((tmp_C_sd3[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd3 = np.concatenate((tmp_C_sd3, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_C_sd4 = np.array([])
            for tt in range(len(event_C_sd4)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd4[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd4 = ttmp
                elif tt == 1 : tmp_C_sd4 = np.concatenate((tmp_C_sd4[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd4 = np.concatenate((tmp_C_sd4, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_C_sd5 = np.array([])
            for tt in range(len(event_C_sd5)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd5[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd5 = ttmp
                elif tt == 1 : tmp_C_sd5 = np.concatenate((tmp_C_sd5[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd5 = np.concatenate((tmp_C_sd5, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_C_sd6 = np.array([])
            for tt in range(len(event_C_sd6)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd6[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd6 = ttmp
                elif tt == 1 : tmp_C_sd6 = np.concatenate((tmp_C_sd6[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd6 = np.concatenate((tmp_C_sd6, ttmp[...,np.newaxis]), axis = 2)
           
            tmp_C_sd7 = np.array([])
            for tt in range(len(event_C_sd7)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd7[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd7 = ttmp
                elif tt == 1 : tmp_C_sd7 = np.concatenate((tmp_C_sd7[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd7 = np.concatenate((tmp_C_sd7, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_C_sd8 = np.array([])
            for tt in range(len(event_C_sd8)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_C_sd8[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_C_sd8 = ttmp
                elif tt == 1 : tmp_C_sd8 = np.concatenate((tmp_C_sd8[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_C_sd8 = np.concatenate((tmp_C_sd8, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_D_sd1 = np.array([])
            for tt in range(len(event_D_sd1)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd1[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd1 = ttmp
                elif tt == 1 : tmp_D_sd1 = np.concatenate((tmp_D_sd1[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd1 = np.concatenate((tmp_D_sd1, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_D_sd2 = np.array([])
            for tt in range(len(event_D_sd2)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd2[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd2 = ttmp
                elif tt == 1 : tmp_D_sd2 = np.concatenate((tmp_D_sd2[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd2 = np.concatenate((tmp_D_sd2, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_D_sd3 = np.array([])
            for tt in range(len(event_D_sd3)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd3[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd3 = ttmp
                elif tt == 1 : tmp_D_sd3 = np.concatenate((tmp_D_sd3[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd3 = np.concatenate((tmp_D_sd3, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_D_sd4 = np.array([])
            for tt in range(len(event_D_sd4)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd4[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd4 = ttmp
                elif tt == 1 : tmp_D_sd4 = np.concatenate((tmp_D_sd4[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd4 = np.concatenate((tmp_D_sd4, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_D_sd5 = np.array([])
            for tt in range(len(event_D_sd5)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd5[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd5 = ttmp
                elif tt == 1 : tmp_D_sd5 = np.concatenate((tmp_D_sd5[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd5 = np.concatenate((tmp_D_sd5, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_D_sd6 = np.array([])
            for tt in range(len(event_D_sd6)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd6[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd6 = ttmp
                elif tt == 1 : tmp_D_sd6 = np.concatenate((tmp_D_sd6[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd6 = np.concatenate((tmp_D_sd6, ttmp[...,np.newaxis]), axis = 2)
           
            tmp_D_sd7 = np.array([])
            for tt in range(len(event_D_sd7)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd7[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd7 = ttmp
                elif tt == 1 : tmp_D_sd7 = np.concatenate((tmp_D_sd7[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd7 = np.concatenate((tmp_D_sd7, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_D_sd8 = np.array([])
            for tt in range(len(event_D_sd8)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_D_sd8[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_D_sd8 = ttmp
                elif tt == 1 : tmp_D_sd8 = np.concatenate((tmp_D_sd8[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_D_sd8 = np.concatenate((tmp_D_sd8, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_B_sd1 = np.array([])
            for tt in range(len(event_B_sd1)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd1[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd1 = ttmp
                elif tt == 1 : tmp_B_sd1 = np.concatenate((tmp_B_sd1[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd1 = np.concatenate((tmp_B_sd1, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_B_sd2 = np.array([])
            for tt in range(len(event_B_sd2)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd2[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd2 = ttmp
                elif tt == 1 : tmp_B_sd2 = np.concatenate((tmp_B_sd2[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd2 = np.concatenate((tmp_B_sd2, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_B_sd3 = np.array([])
            for tt in range(len(event_B_sd3)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd3[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd3 = ttmp
                elif tt == 1 : tmp_B_sd3 = np.concatenate((tmp_B_sd3[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd3 = np.concatenate((tmp_B_sd3, ttmp[...,np.newaxis]), axis = 2)
            
            tmp_B_sd4 = np.array([])
            for tt in range(len(event_B_sd4)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd4[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd4 = ttmp
                elif tt == 1 : tmp_B_sd4 = np.concatenate((tmp_B_sd4[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd4 = np.concatenate((tmp_B_sd4, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_B_sd5 = np.array([])
            for tt in range(len(event_B_sd5)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd5[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd5 = ttmp
                elif tt == 1 : tmp_B_sd5 = np.concatenate((tmp_B_sd5[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd5 = np.concatenate((tmp_B_sd5, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_B_sd6 = np.array([])
            for tt in range(len(event_B_sd6)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd6[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd6 = ttmp
                elif tt == 1 : tmp_B_sd6 = np.concatenate((tmp_B_sd6[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd6 = np.concatenate((tmp_B_sd6, ttmp[...,np.newaxis]), axis = 2)
           
            tmp_B_sd7 = np.array([])
            for tt in range(len(event_B_sd7)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd7[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd7 = ttmp
                elif tt == 1 : tmp_B_sd7 = np.concatenate((tmp_B_sd7[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd7 = np.concatenate((tmp_B_sd7, ttmp[...,np.newaxis]), axis = 2)
                
            tmp_B_sd8 = np.array([])
            for tt in range(len(event_B_sd8)):
                ttmp = np.array([])
                
                tmp_tm = trig[event_B_sd8[tt],od_offset]
                ttmp = data[:, int(tmp_tm) - int(margin_time*srate) : int(tmp_tm) + int(rot*srate) + int(margin_time*srate)]
                
                if tt == 0: tmp_B_sd8 = ttmp
                elif tt == 1 : tmp_B_sd8 = np.concatenate((tmp_B_sd8[...,np.newaxis], ttmp[...,np.newaxis]), axis = 2)
                else : tmp_B_sd8 = np.concatenate((tmp_B_sd8, ttmp[...,np.newaxis]), axis = 2)
            
            
            if ss == 0: 
                epoch_C_sd1 = tmp_C_sd1
                epoch_C_sd2 = tmp_C_sd2
                epoch_C_sd3 = tmp_C_sd3
                epoch_C_sd4 = tmp_C_sd4
                epoch_C_sd5 = tmp_C_sd5
                epoch_C_sd6 = tmp_C_sd6
                epoch_C_sd7 = tmp_C_sd7
                epoch_C_sd8 = tmp_C_sd8
                
                epoch_D_sd1 = tmp_D_sd1
                epoch_D_sd2 = tmp_D_sd2
                epoch_D_sd3 = tmp_D_sd3
                epoch_D_sd4 = tmp_D_sd4
                epoch_D_sd5 = tmp_D_sd5
                epoch_D_sd6 = tmp_D_sd6
                epoch_D_sd7 = tmp_D_sd7
                epoch_D_sd8 = tmp_D_sd8
                
                epoch_B_sd1 = tmp_B_sd1
                epoch_B_sd2 = tmp_B_sd2
                epoch_B_sd3 = tmp_B_sd3
                epoch_B_sd4 = tmp_B_sd4
                epoch_B_sd5 = tmp_B_sd5
                epoch_B_sd6 = tmp_B_sd6
                epoch_B_sd7 = tmp_B_sd7
                epoch_B_sd8 = tmp_B_sd8
                
            else: 
                epoch_C_sd1 = np.concatenate((epoch_C_sd1, tmp_C_sd1), axis=2)
                epoch_C_sd2 = np.concatenate((epoch_C_sd2, tmp_C_sd2), axis=2)
                epoch_C_sd3 = np.concatenate((epoch_C_sd3, tmp_C_sd3), axis=2)
                epoch_C_sd4 = np.concatenate((epoch_C_sd4, tmp_C_sd4), axis=2)
                epoch_C_sd5 = np.concatenate((epoch_C_sd5, tmp_C_sd5), axis=2)
                epoch_C_sd6 = np.concatenate((epoch_C_sd6, tmp_C_sd6), axis=2)
                epoch_C_sd7 = np.concatenate((epoch_C_sd7, tmp_C_sd7), axis=2)
                epoch_C_sd8 = np.concatenate((epoch_C_sd8, tmp_C_sd8), axis=2)
                
                epoch_D_sd1 = np.concatenate((epoch_D_sd1, tmp_D_sd1), axis=2)
                epoch_D_sd2 = np.concatenate((epoch_D_sd2, tmp_D_sd2), axis=2)
                epoch_D_sd3 = np.concatenate((epoch_D_sd3, tmp_D_sd3), axis=2)
                epoch_D_sd4 = np.concatenate((epoch_D_sd4, tmp_D_sd4), axis=2)
                epoch_D_sd5 = np.concatenate((epoch_D_sd5, tmp_D_sd5), axis=2)
                epoch_D_sd6 = np.concatenate((epoch_D_sd6, tmp_D_sd6), axis=2)
                epoch_D_sd7 = np.concatenate((epoch_D_sd7, tmp_D_sd7), axis=2)
                epoch_D_sd8 = np.concatenate((epoch_D_sd8, tmp_D_sd8), axis=2)
                
                epoch_B_sd1 = np.concatenate((epoch_B_sd1, tmp_B_sd1), axis=2)
                epoch_B_sd2 = np.concatenate((epoch_B_sd2, tmp_B_sd2), axis=2)
                epoch_B_sd3 = np.concatenate((epoch_B_sd3, tmp_B_sd3), axis=2)
                epoch_B_sd4 = np.concatenate((epoch_B_sd4, tmp_B_sd4), axis=2)
                epoch_B_sd5 = np.concatenate((epoch_B_sd5, tmp_B_sd5), axis=2)
                epoch_B_sd6 = np.concatenate((epoch_B_sd6, tmp_B_sd6), axis=2)
                epoch_B_sd7 = np.concatenate((epoch_B_sd7, tmp_B_sd7), axis=2)
                epoch_B_sd8 = np.concatenate((epoch_B_sd8, tmp_B_sd8), axis=2)
                
            
            del tmp_C_sd1, tmp_C_sd2, tmp_C_sd3, tmp_C_sd4, tmp_C_sd5, tmp_C_sd6, tmp_C_sd7, tmp_C_sd8
            del tmp_D_sd1, tmp_D_sd2, tmp_D_sd3, tmp_D_sd4, tmp_D_sd5, tmp_D_sd6, tmp_D_sd7, tmp_D_sd8
            del tmp_B_sd1, tmp_B_sd2, tmp_B_sd3, tmp_B_sd4, tmp_B_sd5, tmp_B_sd6, tmp_B_sd7, tmp_B_sd8
            
        fname = os.path.join(fold_epoch_bd,'Sub'+ str(SubNum) + '_' + SubName +'.mat')
        '''
        sio.savemat(fname,
                    {
                        'epoch_C_sd1':epoch_C_sd1, 'epoch_C_sd2':epoch_C_sd2, 
                        'epoch_C_sd3':epoch_C_sd3, 'epoch_C_sd4':epoch_C_sd4, 
                        'epoch_C_sd5':epoch_C_sd5, 'epoch_C_sd6':epoch_C_sd6, 
                        'epoch_C_sd7':epoch_C_sd7, 'epoch_C_sd8':epoch_C_sd8, 
                        
                        'epoch_D_sd1':epoch_D_sd1, 'epoch_D_sd2':epoch_D_sd2, 
                        'epoch_D_sd3':epoch_D_sd3, 'epoch_D_sd4':epoch_D_sd4, 
                        'epoch_D_sd5':epoch_D_sd5, 'epoch_D_sd6':epoch_D_sd6, 
                        'epoch_D_sd7':epoch_D_sd7, 'epoch_D_sd8':epoch_D_sd8, 
                        
                        
                        'epoch_B_sd1':epoch_B_sd1, 'epoch_B_sd2':epoch_B_sd2, 
                        'epoch_B_sd3':epoch_B_sd3, 'epoch_B_sd4':epoch_B_sd4, 
                        'epoch_B_sd5':epoch_B_sd5, 'epoch_B_sd6':epoch_B_sd6, 
                        'epoch_B_sd7':epoch_B_sd7, 'epoch_B_sd8':epoch_B_sd8, 
                        
                        
                        'margin_time':margin_time,
                        'Bad_Ch':Bad_Ch, 'srate':srate
                    })
        '''
        del epoch_C_sd1, epoch_C_sd2, epoch_C_sd3, epoch_C_sd4, epoch_C_sd5, epoch_C_sd6, epoch_C_sd7, epoch_C_sd8
        del epoch_D_sd1, epoch_D_sd2, epoch_D_sd3, epoch_D_sd4, epoch_D_sd5, epoch_D_sd6, epoch_D_sd7, epoch_D_sd8
        del epoch_B_sd1, epoch_B_sd2, epoch_B_sd3, epoch_B_sd4, epoch_B_sd5, epoch_B_sd6, epoch_B_sd7, epoch_B_sd8
        del Bad_Ch, srate
        
        