"""
Author: Jii Kwon <jii.kwon125@gmail.com>
Seoul National University
Human Brain Function Laboratory 

ECoG_Music : Wrapper method
"""

#%% Go to root
import os.path
import os

serv = input("Window:0 Linux:1 -")

if serv == '0':
    serv_name = 'E:\\'
else:
    serv_name = '/mnt/e'
    
folder_std = os.path.join(serv_name, 'ECoG_Music', '2304_Music_Imagery', '2304_Music_Imagery_Decoding_v1')
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

aug_val = None # 1.2 # None or number
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

if aug_val == None: fold_class_FW = os.path.join(folder_std, '14_Clf_FW_cl','sess1-8['+'aug_None_FI]_min_max_train')
else: fold_class_FW = os.path.join(folder_std, '14_Clf_FW_cl','sess1-8['+'aug_'+str(aug_val)+'_FI]_min_max_train')
'''
if serv == 1:
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #plt.rcParams['font.serif'] = plt.rcParams['font.serif'] # 
else: plt.rcParams['font.family'] = 'Times New Roman'    
'''

#%% Analysis
for sub_idx in trange(len(Sub_Nums)):
    if sub_idx == 8: 
        sub_fname, SubName, NumCh, Ch_list, Bad_Ch = basic.info_sub(SubNum = Sub_Nums[sub_idx], file_inform=file_inform)
        
        fold_feature = os.path.join(folder_std, '03_Extracted_Feature')
        fname = os.path.join(fold_feature,'1_D',BASE,sub_fname+'.mat')
        file = sio.loadmat(fname)
        Test_sess = int(file['Test_sess'][0])
        del file, fname, fold_feature
        
        for sess in range(1, Test_sess+1):
            
            #% Load Neural Data
            fname = os.path.join(fold_feature_concat,  sub_fname+'_session_'+str(int(sess)) +'.pkl')
            neural = basic.load_pkl(fname)
            test_data_org, test_label_org = neural['train_data'],neural['train_label']
            
            fn_mat = glob.glob(os.path.join(fold_feature_concat, sub_fname+'_session_'+'*.pkl'))
            filtered_fn_mat = [f for f in fn_mat if not f.endswith(fname)]
            
            train_data_org = pd.DataFrame([])
            train_label_org = pd.DataFrame([])
            for ff in filtered_fn_mat:
                others = basic.load_pkl(ff)
                tmp_data, tmp_label = others['train_data'],others['train_label']
                train_data_org = pd.concat([train_data_org, tmp_data], axis=0, ignore_index=True)
                train_label_org = pd.concat([train_label_org, tmp_label], axis=0, ignore_index=True)
            
            #% Save result : scaling = 'standard'
            train_label_org.columns = ['class']
            train_label_org = train_label_org['class']
            train_label_org = train_label_org.sample(frac=1).reset_index(drop=True)
    
    
            max_values = train_data_org.max()
            min_values = train_data_org.min()
            
            
            train_data_org = ml_models.scaling(train_data_org, method = 'min-max')
            test_data = (test_data_org - min_values) / (max_values - min_values)
    
            train_label_org, orgin_label = ml_models.label_transpose(train_label_org)
            test_label, orgin_tt_label = ml_models.label_transpose(test_label_org)
            chance_level = 1/len(Counter(train_label_org))
            
            if aug_val != None:
                aug_count = int(min(Counter(train_label_org).values())*aug_val)
                train_data, train_label = ml_models.sampling(count=aug_count, dataset=train_data_org, data_label=train_label_org, randstate=random_state)
            else:
                train_data, train_label = train_data_org, train_label_org
                
            test_accuracy = []
            import random
            for per in range(200):
                random.seed(per) #42
                a = random.shuffle(train_label)
                
                sorted_indices, sorted_feature_name = ml_models.order_forFW(train_data, train_label, random_state = random_state)
                
                models = ml_models.set_models(random_state=random_state)
                strat_k_fold = ml_models.strat_kfold(n_splits=kfold, random_state=random_state, shuffle=False)
                
                
                
                
                #cv_result = ml_models.fw_cross_val(train_data, train_label, sorted_indices, models, strat_k_fold, imbalance = aug_val)
                best_model, test_acc, cv_result, predictions = ml_models.evaluate_best_model(train_data, train_label, test_data, test_label, sorted_indices, models, strat_k_fold, imbalance=aug_val)
            
                test_accuracy.append(test_acc)
            
            test_accuracy = np.mean(np.array(test_accuracy))
            #% Save result
            fold_name = basic.set_folder(os.path.join(fold_class_FW, 'scaled_min-max_sess'+str(int(sess)),sub_fname))
            ml_models.plot_cv_result(chance_level, cv_result, sorted_indices, test_accuracy, fold_name)
           
            fname = os.path.join(os.path.join(fold_class_FW, 'scaled_min-max_sess'+str(int(sess))), sub_fname +'.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'cv_result': cv_result, 'sorted_feature_name':sorted_feature_name, 'sorted_indices':sorted_indices, 'predictions':predictions,'test_acc':test_accuracy}, f)
    
            fname = os.path.join(os.path.join(fold_class_FW, 'scaled_min-max_sess'+str(int(sess))), 'model_'+sub_fname +'.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'cv_result': cv_result, 'sorted_feature_name':sorted_feature_name, 'sorted_indices':sorted_indices, 'best_model':best_model}, f)
    
    
            print('\nmin_max_sess'+str(int(sess))+'-'+str(np.round(test_accuracy,3)*100))
            
    

