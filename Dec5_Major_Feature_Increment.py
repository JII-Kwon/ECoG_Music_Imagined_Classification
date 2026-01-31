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
from utils_basic_music import np, plt, sio, pd, sns
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

aug_val = 1.5 #1.2 # None or number
param_set = None #Boolean : getting the gridsearch result
scaling = 'min-max'

adj_fold = 'v231108_sess1-8[aug_1.5_FI]_min_max_train'

#%% Setting Analysis Basic Parameter [not necessaries of adjusting param]
# Basic Param
Sub_Nums = np.arange(1,11)
bands = ['D', 'T', 'A', 'B', 'G', 'HG']
band_frqs = [(1,4), (4,7), (8,12), (12,30), (30,59), (61,150)]

file_inform = os.path.join(serv_name, 'ECoG_Music', 'Subject_info.mat')

# Set folder name
fold_select = os.path.join(folder_std, '01_select_electrode', 'case2_P1orP2')

fold_class_FW = os.path.join(folder_std, '04_Clf_FW', adj_fold)
fold_acc = basic.set_folder(os.path.join(folder_std, '05_Resul_accuracy'))
fold_major = basic.set_folder(os.path.join(folder_std, '07_Major_Feature_Distribution_v240723','Increment_Value', adj_fold))

if serv == 1: plt.rcParams['font.family'] = 'DejaVu Serif'
else: plt.rcParams['font.family'] = 'Times New Roman'    

#%% Analysis
# 1st stage: 각 영역당 몇개의 channel이 있는지
total_elec = pd.DataFrame([])
for sub_idx in trange(len(Sub_Nums)):
    sub_fname, SubName, NumCh, Ch_list, Bad_Ch = basic.info_sub(SubNum = Sub_Nums[sub_idx], file_inform=file_inform)
    
    info_elec = pd.read_csv(os.path.join(fold_select, sub_fname+'.csv'), index_col=0)
    info_elec = pd.DataFrame(info_elec['Name (aparc)'])
    total_elec = pd.concat((total_elec,info_elec), axis = 0, ignore_index=True)

electrode_count = pd.DataFrame(total_elec['Name (aparc)'].value_counts())
electrode_count.to_excel(os.path.join(fold_major, 'total_implanted_elec.xlsx'))
#%%
major_feature = pd.DataFrame([])
for sub_idx in trange(len(Sub_Nums)):
    sub_fname, SubName, NumCh, Ch_list, Bad_Ch = basic.info_sub(SubNum = Sub_Nums[sub_idx], file_inform=file_inform)
    
    clf_result = os.path.join(fold_acc, adj_fold+'.xlsx')
    clf_result = pd.read_excel(clf_result, sheet_name=sub_fname, index_col=0)
    
    train_acc = max(clf_result['train.1'])
    idx_max = clf_result[scaling].idxmax()
    best_model_name = clf_result['model.1'].iloc[idx_max]
    
    subfolders = basic.find_names(fold_class_FW, pattern = 'scaled_'+ scaling +'_sess', type_='dir')
    clf_max = subfolders[idx_max]
    
    clf_max = basic.load_pkl(os.path.join(fold_class_FW, clf_max,sub_fname+'.pkl'))
    test_acc = clf_max['test_acc']
    cv_result = clf_max['cv_result']
    cv_result = cv_result[best_model_name]
    
    chance_level = 1/6
    
    abs_increment = []
    abs_increment.append(cv_result[0][0]-chance_level)
    
    for ii in range(1,len(cv_result)):
        iii = cv_result[ii][0]-cv_result[ii-1][0]
        
        abs_increment.append(iii)
    
    rel_increment = []
    for num, value in enumerate(abs_increment):
        rel_increment.extend([value/(num+1)])
    
    
    fin_increment = []
    for idx in range(clf_result.loc[idx_max, 'indx.1']+1):
        fin_increment.append(sum(rel_increment[idx:clf_result.loc[idx_max, 'indx.1']+1]))
    
    sub_major_feat = clf_max['sorted_feature_name']
    sub_major_feat = sub_major_feat[:clf_result.loc[idx_max, 'indx.1']+1]
    
    sub_major_feat = pd.DataFrame(sub_major_feat, columns = ['Feature'])
    sub_major_feat[['Band', 'ChNum']] = sub_major_feat['Feature'].str.extract(r'(.*)\_Ch(\d+)')
    
    
    info_elec = pd.read_csv(os.path.join(fold_select, sub_fname+'.csv'), index_col=0)
    info_elec = pd.DataFrame(info_elec['Name (aparc)'])
    
    extract_Ch = []
    extract_Ch = [info_elec.loc[np.where(Ch_list == int(i))[0][0]].values[0] for i in sub_major_feat['ChNum']]
    extract_Ch = pd.DataFrame(extract_Ch, columns=['Area'])
    extract_Ch = pd.concat((extract_Ch, sub_major_feat['Band']), axis = 1)
    
    extract_Ch['Merged'] = extract_Ch['Area'] + '-' + extract_Ch['Band']
    
    extract_Ch['increment'] = pd.DataFrame(fin_increment)
    
    major_feature = pd.concat((major_feature,extract_Ch), axis = 0, ignore_index=True)

major_feature.to_excel(os.path.join(fold_major, 'Total_result.xlsx'), index=False)

#%%
# 사람들이 

#%%
import matplotlib.colors as mcolors


processed = pd.read_excel(os.path.join(fold_major, 'Final.xlsx'))
processed['Feature'] = processed['Feature'].str.replace('-', ' ')

sum_processed = processed.sort_values(by='Sum', ascending=False)
threshold = sum_processed['Sum'].quantile(0.9)

df_top = sum_processed[sum_processed['Sum'] >= threshold]
norm = mcolors.Normalize(vmin=df_top['Implant Count'].min(), vmax=df_top['Implant Count'].max())
tt = pd.DataFrame([' ', ' ',' ',' '])
# Choose a colormap
colormap = plt.cm.Blues
colors = colormap(norm(df_top['Implant Count']))
colors = np.concatenate((colors, np.full((len(tt), 4), [0.8, 0.8, 0.8, 1])))

all_features = pd.concat([df_top['Feature'], tt[0]], axis=0, ignore_index=True)
all_sums = pd.concat([df_top['Sum']*100, pd.Series([0, 0, 0, 0])], axis=0)

fig, ax = plt.subplots(figsize=(20, 12))

ax.barh(all_features, all_sums, color=colors, edgecolor='black')
ax.set_xlabel('Increment value', fontsize=20)
ax.set_ylabel('Feature', fontsize=20)
#ax.set_title('Top 20% of Sum Values with Different Bar Colors')
ax.invert_yaxis()  # To display the highest values at the top
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.tick_params(axis='both', labelsize=22)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Implant Count')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('# of Electrode Implant', fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(fold_major, 'Increment_Value.png'), dpi = 300)
plt.show()

#%%
average = processed.sort_values(by='Average', ascending=False)

threshold = average['Average'].quantile(0.8)
df_top = average[average['Average'] >= threshold]
norm = mcolors.Normalize(vmin=df_top['Implant Count'].min(), vmax=df_top['Implant Count'].max())

# Choose a colormap
colormap = plt.cm.Blues

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create a horizontal bar plot
ax.barh(df_top['Feature'], df_top['Average'], color=colormap(norm(df_top['Implant Count'])), edgecolor='black')
ax.set_xlabel('Average')
ax.set_ylabel('Feature')
ax.set_title('Top 20% of Sum Values with Different Bar Colors')
ax.invert_yaxis()  # To display the highest values at the top
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Implant Count')

plt.show()

#%% All data


import matplotlib.colors as mcolors


processed = pd.read_excel(os.path.join(fold_major, 'Final.xlsx'))
processed['Feature'] = processed['Feature'].str.replace('-', ' ')

sum_processed = processed.sort_values(by='Sum', ascending=False)
threshold = sum_processed['Sum'].quantile(0.9)

df_top = sum_processed#[sum_processed['Sum']]
norm = mcolors.Normalize(vmin=df_top['Implant Count'].min(), vmax=df_top['Implant Count'].max())
#tt = pd.DataFrame([' ', ' ',' ',' '])
# Choose a colormap
colormap = plt.cm.Blues
colors = colormap(norm(df_top['Implant Count']))
colors = np.concatenate((colors, np.full((len(tt), 4), [0.8, 0.8, 0.8, 1])))

all_features = df_top['Feature']
#all_features = pd.concat([df_top['Feature'], tt[0]], axis=0, ignore_index=True)
all_sums = df_top['Sum']*100
#all_sums = pd.concat([df_top['Sum']*100, pd.Series([0, 0, 0, 0])], axis=0)

for ii in range(int(138/23)):
    fig, ax = plt.subplots(figsize=(22, 25))
    
    ax.barh(all_features[ii*23:(1+ii)*23], all_sums[ii*23:(1+ii)*23], color=colors, edgecolor='black')
    ax.set_xlabel('Increment value', fontsize=24)
    ax.set_ylabel('Feature', fontsize=24)
    #ax.set_title('Top 20% of Sum Values with Different Bar Colors')
    ax.invert_yaxis()  # To display the highest values at the top
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=24)
    ax.tick_params(axis='y', labelsize=28) 
    plt.xlim([0, 20])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Implant Count')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('# of Electrode Implant', fontsize=24)
    plt.tight_layout()
    
    plt.savefig(os.path.join(fold_major, 'Increment_Value'+str(ii)+'.png'), dpi = 300)
    plt.show()