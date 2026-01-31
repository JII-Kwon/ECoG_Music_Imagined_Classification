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
import pickle
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
fold_epoch = os.path.join(folder_std, '04_Decoding_melody', 'session_answer')
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
for sub_idx in trange(len(Sub_Nums)):
    SubNum = Sub_Nums[sub_idx]
    
    SubName, NumCh, Ch_list, Bad_Ch = info_sub(SubNum)
    SubToken = 'Sub' + str(SubNum) + '_' + SubName
    
    fn_mat = glob.glob(os.path.join(fold_mat, '1_D', '1_D_' + SubToken+'_sess*.mat'))

    for ss, fn in enumerate(fn_mat):
        sess = ss +1
        
        fold_epoch_ss = os.path.join(fold_epoch, 'session_'+str(sess))
        if not os.path.exists(fold_epoch_ss): os.makedirs(fold_epoch_ss)
        
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
        
        trig = trig.apply(pd.Series.explode)
        
        time_flow = []
        
        th_song_a = []
        th_song_b = []
        a_tf = 0
        b_tf = 0
        for ii in range(trig.shape[0]):
            tmp_task = int(trig.loc[ii, 'stimtype'])
            
            if a_tf >= 1 : a_tf += 1
            if b_tf >= 1 : b_tf += 1
            if (tmp_task == 2) :
                if int(trig.loc[ii-9, 'stimtype']) == 9: a_tf = 1
            if (tmp_task == 4) : a_tf = 0             
            if (tmp_task == 2):
                if int(trig.loc[ii-9, 'stimtype']) == 3: b_tf = 1
            if (tmp_task == 4) : b_tf = 0
                
            th_song_a.append(a_tf)
            th_song_b.append(b_tf)
            
        
        trig['th_song_a'] = pd.DataFrame(th_song_a)
        trig['th_song_b'] = pd.DataFrame(th_song_b)
        
        th_song =[]
        for ii in range(trig.shape[0]):
            tmp_mj = int(trig.loc[ii, 'major'])
            if tmp_mj == 1: tmp_mj = 'C'
            elif tmp_mj == 2: tmp_mj = 'D'
            elif tmp_mj == 3: tmp_mj = 'B'
            
            if int(trig.loc[ii, 'song']) == 0: tmp_sg = 'NONE'
            else: tmp_sg = 'sg' + str(int(trig.loc[ii, 'song']))
            
            if tmp_sg == 'NONE': th_song.append(tmp_sg)
            elif int(trig.loc[ii, 'th_song_a']) >= 1 : th_song.append(tmp_mj+'_'+tmp_sg+'_1_'+str(int(trig.loc[ii, 'th_song_a'])))
            elif int(trig.loc[ii, 'th_song_b']) >= 1 : th_song.append(tmp_mj+'_'+tmp_sg+'_2_'+str(int(trig.loc[ii, 'th_song_b'])))
            else: th_song.append('NONE')
        
        trig['answer'] = pd.DataFrame(th_song)
        
        trig_header = trig.columns.tolist()     
        
        trig = pd.DataFrame.to_numpy(trig)
    
        od_stim = trig_header.index('stimtype') #1: Music listening 2: Imagery 3: Production
        od_task = trig_header.index(standard)
        od_maj = trig_header.index('major')
        od_offset = trig_header.index('offset')
        time_flow = trig_header.index('answer')
        
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
        
      
        ans_C_sd1 = []
        for tt in range(len(event_C_sd1)):
            tmp_tm = trig[event_C_sd1[tt],time_flow]
            ans_C_sd1.append(tmp_tm)
        
        ans_C_sd2 = []
        for tt in range(len(event_C_sd2)):
            tmp_tm = trig[event_C_sd2[tt],time_flow]
            ans_C_sd2.append(tmp_tm)
            
        ans_C_sd3 = []
        for tt in range(len(event_C_sd3)):
            tmp_tm = trig[event_C_sd3[tt],time_flow]
            ans_C_sd3.append(tmp_tm)
        
        ans_C_sd4 = []
        for tt in range(len(event_C_sd4)):
            tmp_tm = trig[event_C_sd4[tt],time_flow]
            ans_C_sd4.append(tmp_tm)
            
        ans_C_sd5 = []
        for tt in range(len(event_C_sd5)):
            tmp_tm = trig[event_C_sd5[tt],time_flow]
            ans_C_sd5.append(tmp_tm)
        
        ans_C_sd6 = []
        for tt in range(len(event_C_sd6)):
            tmp_tm = trig[event_C_sd6[tt],time_flow]
            ans_C_sd6.append(tmp_tm)
            
        ans_C_sd7 = []
        for tt in range(len(event_C_sd7)):
            tmp_tm = trig[event_C_sd7[tt],time_flow]
            ans_C_sd7.append(tmp_tm)
            
        ans_C_sd8 = []
        for tt in range(len(event_C_sd8)):
            tmp_tm = trig[event_C_sd8[tt],time_flow]
            ans_C_sd8.append(tmp_tm)
        
        ans_D_sd1 = []
        for tt in range(len(event_D_sd1)):
            tmp_tm = trig[event_D_sd1[tt],time_flow]
            ans_D_sd1.append(tmp_tm)
        
        ans_D_sd2 = []
        for tt in range(len(event_D_sd2)):
            tmp_tm = trig[event_D_sd2[tt],time_flow]
            ans_D_sd2.append(tmp_tm)
            
        ans_D_sd3 = []
        for tt in range(len(event_D_sd3)):
            tmp_tm = trig[event_D_sd3[tt],time_flow]
            ans_D_sd3.append(tmp_tm)
        
        ans_D_sd4 = []
        for tt in range(len(event_D_sd4)):
            tmp_tm = trig[event_D_sd4[tt],time_flow]
            ans_D_sd4.append(tmp_tm)
            
        ans_D_sd5 = []
        for tt in range(len(event_D_sd5)):
            tmp_tm = trig[event_D_sd5[tt],time_flow]
            ans_D_sd5.append(tmp_tm)
        
        ans_D_sd6 = []
        for tt in range(len(event_D_sd6)):
            tmp_tm = trig[event_D_sd6[tt],time_flow]
            ans_D_sd6.append(tmp_tm)
            
        ans_D_sd7 = []
        for tt in range(len(event_D_sd7)):
            tmp_tm = trig[event_D_sd7[tt],time_flow]
            ans_D_sd7.append(tmp_tm)
            
        ans_D_sd8 = []
        for tt in range(len(ans_D_sd8)):
            tmp_tm = trig[ans_D_sd8[tt],time_flow]
            ans_D_sd8.append(tmp_tm)
        
        ans_B_sd1 = []
        for tt in range(len(event_B_sd1)):
            tmp_tm = trig[event_B_sd1[tt],time_flow]
            ans_B_sd1.append(tmp_tm)
        
        ans_B_sd2 = []
        for tt in range(len(event_B_sd2)):
            tmp_tm = trig[event_B_sd2[tt],time_flow]
            ans_B_sd2.append(tmp_tm)
            
        ans_B_sd3 = []
        for tt in range(len(event_B_sd3)):
            tmp_tm = trig[event_B_sd3[tt],time_flow]
            ans_B_sd3.append(tmp_tm)
        
        ans_B_sd4 = []
        for tt in range(len(event_B_sd4)):
            tmp_tm = trig[event_B_sd4[tt],time_flow]
            ans_B_sd4.append(tmp_tm)
            
        ans_B_sd5 = []
        for tt in range(len(event_B_sd5)):
            tmp_tm = trig[event_B_sd5[tt],time_flow]
            ans_B_sd5.append(tmp_tm)
        
        ans_B_sd6 = []
        for tt in range(len(event_B_sd6)):
            tmp_tm = trig[event_B_sd6[tt],time_flow]
            ans_B_sd6.append(tmp_tm)
            
        ans_B_sd7 = []
        for tt in range(len(event_B_sd7)):
            tmp_tm = trig[event_B_sd7[tt],time_flow]
            ans_B_sd7.append(tmp_tm)
            
        ans_B_sd8 = []
        for tt in range(len(ans_B_sd8)):
            tmp_tm = trig[ans_B_sd8[tt],time_flow]
            ans_B_sd8.append(tmp_tm)
                
        final_list = []
        for i in range(1, 7):
            for prefix in ['C', 'D', 'B']:
                list_name = f'ans_{prefix}_sd{i}'
                final_list.extend(eval(list_name))
                    
        
        fname = os.path.join(fold_epoch, 'session_'+str(sess),'Sub'+ str(SubNum) + '_' + SubName +'.pkl')
        with open(fname, 'wb') as f:
            pickle.dump({'ans_C_sd1': ans_C_sd1, 'ans_C_sd2': ans_C_sd2, 'ans_C_sd3': ans_C_sd3,
                         'ans_C_sd4': ans_C_sd4, 'ans_C_sd5': ans_C_sd5, 'ans_C_sd6': ans_C_sd6,
                         'ans_C_sd7': ans_C_sd7, 'ans_C_sd8': ans_C_sd8, 
                         'ans_D_sd1': ans_D_sd1, 'ans_D_sd2': ans_D_sd2, 'ans_D_sd3': ans_D_sd3,
                         'ans_D_sd4': ans_D_sd4, 'ans_D_sd5': ans_D_sd5, 'ans_D_sd6': ans_D_sd6,
                         'ans_D_sd7': ans_D_sd7, 'ans_D_sd8': ans_D_sd8, 
                         'ans_B_sd1': ans_B_sd1, 'ans_B_sd2': ans_B_sd2, 'ans_B_sd3': ans_B_sd3,
                         'ans_B_sd4': ans_B_sd4, 'ans_B_sd5': ans_B_sd5, 'ans_B_sd6': ans_B_sd6,
                         'ans_B_sd7': ans_B_sd7, 'ans_B_sd8': ans_B_sd8, 
                         'ans_tot': final_list
                         }, f)
        del ans_C_sd1, ans_C_sd2, ans_C_sd3, ans_C_sd4, ans_C_sd5, ans_C_sd6, ans_C_sd7, ans_C_sd8
        del ans_D_sd1, ans_D_sd2, ans_D_sd3, ans_D_sd4, ans_D_sd5, ans_D_sd6, ans_D_sd7, ans_D_sd8
        del ans_B_sd1, ans_B_sd2, ans_B_sd3, ans_B_sd4, ans_B_sd5, ans_B_sd6, ans_B_sd7, ans_B_sd8
        
        
        fname = os.path.join(fold_epoch, 'session_'+str(sess),'Sub'+ str(SubNum) + '_' + SubName +'.pkl')
        with open(fname, 'rb') as f:
           df = pickle.load(f)