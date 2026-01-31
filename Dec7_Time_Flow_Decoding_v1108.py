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
    
folder_std = os.path.join(serv_name, 'ECoG_Music', '2304_Music_Imagery_Decoding_v1')
folder_scripts = os.path.join(folder_std, '00_scripts')
    
os.chdir(folder_scripts)
os.getcwd()

#%% import library
from utils_basic_music import np, plt, sio, pd, sns
from utils_basic_music import basic
from utils_etc_function import Utils 
from utils_ml_model import ml_models
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
from collections import Counter
import pickle
import glob
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

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

adj_fold ='v231108_sess1-8[aug_1.5_FI]_min_max_train'

#%%
answer_sg1_1 = ['SD1','SD1','SD2','SD3','SD1']
answer_sg2_1 = ['SD4','SD4','SD3','SD3','SD2','SD2','SD1']
answer_sg2_2 = ['SD5','SD5','SD4','SD4','SD3','SD3','SD2']
answer_sg3_1 = ['SD3','SD1','SD3','SD5','SD6','SD6','SD6']
answer_sg4_2 = ['SD5','SD4','SD3','SD2','SD1']
answer_sg5_2 = ['SD2','SD2','SD2','SD3','SD2']

answer_sg1_1 = [0,0,1,2,0]
answer_sg2_1 = [3,3,2,2,1,1,0]
answer_sg2_2 = [4,4,3,3,2,2,1]
answer_sg3_1 = [2,2,4,5,5,5]
answer_sg4_2 = [4,3,2,1,0]
answer_sg5_2 = [1,1,1,2,1]
answers_dict = {
    'sg1_1': answer_sg1_1,
    'sg2_1': answer_sg2_1,
    'sg2_2': answer_sg2_2,
    'sg3_1': answer_sg3_1,
    'sg4_2': answer_sg4_2,
    'sg5_2': answer_sg5_2
}

#%%
def cos_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Check for zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0
    
    # Calculate cosine similarity
    distance_cos = np.dot(a, b) / (norm_a * norm_b)
    return distance_cos

#%% Setting Analysis Basic Parameter [not necessaries of adjusting param]
# Basic Param
Sub_Nums = np.arange(1,11)
bands = ['D', 'T', 'A', 'B', 'G', 'HG']
band_frqs = [(1,4), (4,7), (8,12), (12,30), (30,59), (61,150)]

file_inform = os.path.join(serv_name, 'ECoG_Music', 'Subject_info.mat')

# Set folder name
fold_class_FW = os.path.join(folder_std, '04_Clf_FW', adj_fold)
fold_acc = basic.set_folder(os.path.join(folder_std, '05_Resul_accuracy'))
fold_melody = basic.set_folder(os.path.join(folder_std, '06_Decoding_melody'))
fold_feature = os.path.join(folder_std, '06_Decoding_melody', '01_Extracted_Feature', 'concat_' + BASE)
fold_melody_train = basic.set_folder(os.path.join(folder_std, '06_Decoding_melody', '02_Train_model'))

fold_train = os.path.join(folder_std, '03_Extracted_Feature','concat_session_' + BASE)
fold_melody_answer = basic.set_folder(os.path.join(folder_std, '06_Decoding_melody', '02_session_answer'))
fold_result = basic.set_folder(os.path.join(folder_std, '06_Decoding_melody', '03_Result'))
#%% Analysis
sub_tot = [] 
for sub_idx in trange(len(Sub_Nums)):
    sub_fname, SubName, NumCh, Ch_list, Bad_Ch = basic.info_sub(SubNum = Sub_Nums[sub_idx], file_inform=file_inform)
    
    clf_result = os.path.join(fold_acc, adj_fold+'.xlsx')
    clf_result = pd.read_excel(clf_result, sheet_name=sub_fname, index_col=0)
    
    idx_max = clf_result[scaling].idxmax()
    
    subfolders = basic.find_names(fold_class_FW, pattern = 'scaled_'+ scaling +'_sess', type_='dir')
    max_info = subfolders[idx_max]
    max_info = basic.load_pkl(os.path.join(fold_class_FW, max_info,sub_fname+'.pkl'))
    
    major_feat = max_info['sorted_feature_name']
    major_feat = major_feat[:clf_result.loc[idx_max, 'indx.1']+1]
    major_feat_ind = clf_result.loc[idx_max, 'indx.1']
    
    sorted_indices = max_info['sorted_indices']
    
    test_predict = max_info['predictions']
    
    fname = os.path.join(fold_melody_answer, 'session_' + str(idx_max+1), sub_fname + '.pkl')
    answer = basic.load_pkl(fname)
    answer = answer['ans_tot']
    
    result_pd = pd.DataFrame(test_predict, columns=['predict'])
    #result_pd['answer'] = pd.DataFrame(test_label)
    result_pd['song'] = pd.DataFrame(answer)

    result_pd['prefix'] = result_pd['song'].str.extract(r'([A-Z]_sg\d_\d)')
    
    grouped = result_pd.groupby('prefix')
    trials = []
    for prefix, group_df in grouped:
        exec(prefix + '= group_df')
        trials.append(prefix)
    
    tt_ans = []
    r_each_song = []
    cos_each_song = []
    
    ans_tot = []
    song_list = []
    tot_ans = pd.DataFrame([])
    for prefix in trials:
        exec('tt_ans = ' + prefix)
        if tt_ans.shape[0] < 5: continue
        if (prefix =='B_sg4_1') |(prefix =='D_sg4_1')| (prefix =='C_sg4_1') : continue
    
        parts = prefix.split('_')
        extracted = '_'.join(parts[1:]) 
        
        answer = answers_dict.get(extracted, None) 
        
        tt_ans_sort  = tt_ans.sort_values(by='song')
        r, p = spearmanr(np.array(tt_ans_sort['predict']), np.array(answer))
        sim = cos_sim(np.array(tt_ans_sort['predict']), np.array(answer))
        
        r_each_song.append(r)
        cos_each_song.append(sim)        
        tot_ans = pd.concat((tot_ans, tt_ans_sort), axis = 0)
        song_list.append(prefix)
        
        ans_tot.extend(answer)
                
    r_total, p = spearmanr(np.array(tot_ans['predict']), np.array(ans_tot))
    sim_tot = cos_sim(np.array(tot_ans['predict']), np.array(ans_tot))
    
    if r_total > 0.5:
        plt.plot(np.arange(len(np.array(tot_ans['predict']))),np.array(tot_ans['predict']))
        plt.plot(np.arange(len(np.array(tot_ans['predict']))), ans_tot)
    
    sub_tot.append(np.array(tot_ans['predict']))
    
    song_result = pd.concat((pd.DataFrame(song_list),pd.DataFrame(r_each_song), pd.DataFrame(cos_each_song)), axis = 1)
    song_result.columns = ['song', 'spearmanr', 'cosine']
    
    
    control_r_each_song = []
    control_cos_each_song = []
    
    for prefix in trials:
        exec('tt_ans = ' + prefix)
        if tt_ans.shape[0] < 5: continue
        if (prefix =='B_sg4_1') |(prefix =='D_sg4_1')| (prefix =='C_sg4_1') : continue
    
        parts = prefix.split('_')
        extracted = '_'.join(parts[1:]) 
        
        answer = answers_dict.get(extracted, None) 
        
        for ii in range(1000):
            tt_ans_sort  = tt_ans.sort_values(by='song')
            rand = np.array(tt_ans_sort['predict'])
            np.random.shuffle(rand)
            
            r_rand, p = spearmanr(rand, np.array(answer))
            sim_rand = cos_sim(rand, np.array(answer))
            
            control_r_each_song.append(r_rand)
            control_cos_each_song.append(sim_rand)

    control2_r_each_song = []
    control2_cos_each_song = []
    for prefix in trials:
        exec('tt_ans = ' + prefix)
        if tt_ans.shape[0] < 5: continue
        if (prefix =='B_sg4_1') |(prefix =='D_sg4_1')| (prefix =='C_sg4_1') : continue
    
        parts = prefix.split('_')
        extracted = '_'.join(parts[1:]) 
        
        answer = answers_dict.get(extracted, None) 
        
        for ii in range(1000):
            tt_ans_sort  = tt_ans.sort_values(by='song')
            size = len(tt_ans_sort['predict'])
            
            random_vector = np.random.randint(1, 7, size)
            #np.random.shuffle(rand)
            
            r_rand, p = spearmanr(random_vector, answer)
            sim_rand = cos_sim(random_vector, answer)
            
            control2_r_each_song.append(r_rand)
            control2_cos_each_song.append(sim_rand)
        

    fname = os.path.join(fold_result, sub_fname +'.pkl')
    with open(fname, 'wb') as f:
        pickle.dump({'control_r_each_song': control_r_each_song, 'control_cos_each_song': control_cos_each_song, 
                     'song_result': song_result, 'r_total': r_total, 'sim_tot': sim_tot,
                     }, f)
    
    from scipy.stats import mannwhitneyu
    stat, p_r = mannwhitneyu(control_r_each_song, r_each_song, alternative='two-sided')
    
    
    max_iterations = 100000
    iterations = 0
    
    while iterations < max_iterations:
        random_values = np.random.choice(control_cos_each_song, size=100, replace=True)
        stat, p_sim = mannwhitneyu(random_values, cos_each_song, alternative='less')
        if p_sim < 0.05:
            fold_fig_sim = basic.set_folder(os.path.join(fold_result, 'cosine'))
            
            fname_fig = os.path.join(fold_fig_sim, sub_fname+'.png')
            plt.hist(random_values, alpha=0.5, color='gray', label='random', bins=20, density=True)
            plt.hist(cos_each_song, alpha=0.5, color='b', label='prediction result', bins=20, density=True)
            plt.legend()
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            fname_fig = os.path.join(fold_fig_sim, sub_fname+'_box.png')
            
            max_length = max(len(random_values), len(r_each_song))
            data = pd.DataFrame({
                'random': np.pad(random_values, (0, max_length - len(random_values)), constant_values=np.nan),
                'prediction result': np.pad(cos_each_song, (0, max_length - len(cos_each_song)), constant_values=np.nan)
            })
            data_melted = data.melt(var_name='Group', value_name='cosine similarity')
            
            plt.figure(figsize=(10, 8)) 
            sns.violinplot(
            x='Group',
            y='cosine similarity',
            data=data_melted,
            palette={'random': 'lightgray', 'prediction result': 'lightskyblue'},
            scale='width'  # This scales the violins by the number of observations (narrower = fewer points)
            , cut=0.9)
            plt.xlabel('')
            sns.despine()
            plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            break
        iterations += 1
    
    
    while iterations < max_iterations:
        random_values = np.random.choice(control_r_each_song, size=100, replace=True)
        stat, p_r = mannwhitneyu(random_values, r_each_song, alternative='less')
        if p_r < 0.05:
            
            fold_fig_r = basic.set_folder(os.path.join(fold_result, 'r'))
            
            fname_fig = os.path.join(fold_fig_r, sub_fname+'.png')
            plt.hist(random_values, alpha=0.5, color='gray', label='random', bins=20, density=True)
            plt.hist(r_each_song, alpha=0.5, color='r', label='prediction result', bins=20, density=True)
            plt.legend()
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
            plt.close()
            fname_fig = os.path.join(fold_fig_r, sub_fname+'_box.png')
            
            max_length = max(len(random_values), len(r_each_song))
            data = pd.DataFrame({
                'random': np.pad(random_values, (0, max_length - len(random_values)), constant_values=np.nan),
                'prediction result': np.pad(r_each_song, (0, max_length - len(r_each_song)), constant_values=np.nan)
            })
            
            data_melted = data.melt(var_name='Group', value_name='correlation coefficient')
            
            plt.figure(figsize=(10, 8)) 
            sns.violinplot(
            x='Group',
            y='correlation coefficient',
            data=data_melted,
            palette={'random': 'lightgray', 'prediction result': 'lightcoral'},
            scale='width'  # This scales the violins by the number of observations (narrower = fewer points)
            , cut=0.9)
            plt.xlabel('')
            sns.despine()
            plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            break
        iterations += 1


stacked_data = np.stack(sub_tot)
reshaped_data = stacked_data.reshape((10, 3, 35))
rr_data = np.median(reshaped_data, axis = 1)
mean_values = np.mean(rr_data, axis=0)
std_values = np.std(stacked_data, axis=0)
sem_values = std_values / np.sqrt(stacked_data.shape[0])

def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

stacked_answer = [] 

total_r = []
kk = 0
for kt, ii in enumerate(answers_dict):
    
    i_answer = answers_dict[ii]
    stacked_answer.extend(i_answer)
    
    i_len = len(i_answer)
    i_answer_norm = min_max_normalize(i_answer)
    i_predic_norm = min_max_normalize(mean_values[kk:kk+i_len])
    r,p = spearmanr(i_predic_norm,i_answer_norm)   
    
    #print(r)
    plt.plot(min_max_normalize(i_predic_norm))
    plt.plot(min_max_normalize(i_answer_norm))
    plt.show()
    plt.close()
    kk += i_len
    
    total_r.append(r)

familiarity = [2.636364, 4.909091, 4.909091, 2.454545, 3.636364, 2.363636]

r, p = spearmanr(total_r, familiarity)

total_r2 = [0.22360679774997894,  0.8797804965597554,  0.6575201605867647,  0.49377071987869414, 0.0]
familiarity2 = [2.636364, 4.909091, 4.909091, 2.454545, 2.363636]
r2, p = spearmanr(total_r2, familiarity2)

total_r2 = [0.0, 0.49377071987869414,  0.22360679774997894, 0.6575201605867647,  0.8797804965597554]
familiarity2 = [2.363636, 2.454545, 2.636364, 4.909091, 4.909091]




plt.rcParams['font.family'] = 'Times New Roman'    

data = {'total_r2': total_r2, 'familiarity2': familiarity2}
df = pd.DataFrame(data)
df['category'] = df['familiarity2'].apply(lambda x: '<3' if x < 3 else '>=3')
boxplot_data = [df[df['category'] == '<3']['total_r2'], df[df['category'] == '>=3']['total_r2']]


bp = plt.boxplot(boxplot_data, labels=['low score', 'high score'], patch_artist=True)

for box in bp['boxes']:
    # Change outline color
    box.set(color='midnightblue', linewidth=2)
    # Change fill color
    box.set(facecolor='midnightblue')
    
for median in bp['medians']:
    median.set(color='lightgrey', linewidth=2)
    
plt.xlabel('Familiarity score', fontsize=14)
plt.ylabel('Melody Prediction performance', fontsize=14)

fname_fig = os.path.join(fold_result, 'Familiarity_Compare_box.png')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
[label.set_fontsize(14) for label in ax.get_xticklabels()]
plt.savefig(fname_fig, dpi= 300)
plt.show()
plt.close()

kk = 0
predict = np.array([])
predict_std = np.array([])
predict_ste = np.array([])
answers = np.array([])
kk_tot = [0]
#fig, axs = plt.subplots(1, 6, figsize=(15, 3))  # '5' for five subplots in a row
for kt, ii in enumerate(answers_dict):
    if ii == 'sg4_2': continue
    i_answer = answers_dict[ii]
    stacked_answer.extend(i_answer)
    
    i_len = len(i_answer)
    i_answer_norm = min_max_normalize(i_answer)
    i_predic_norm = min_max_normalize(mean_values[kk:kk+i_len])
    
    predict = np.concatenate((predict, i_predic_norm, [0]))
    answers = np.concatenate((answers, i_answer_norm, [0]))
    
    std_values = np.std(rr_data[:, kk:kk+i_len], axis=0)
    sem_values = std_values / np.sqrt(stacked_data.shape[0])
    
    predict_ste = np.concatenate((predict_ste, sem_values, [0]))
    
    # Plot each array on its own subplot
    #axs[kt].plot(i_answer_norm)
    #axs[kt].plot(i_predic_norm)
    
    kk += i_len
    kk_tot.append(kk)
    
# Set the spacing between subplots
plt.plot(answers, '--k', alpha = 1, linewidth=1.17, label = 'Original melody')
plt.plot(predict, '--', color='tab:red',  alpha = 1, linewidth=1.5, label = 'Reconstructed melody')
plt.fill_between(np.arange(len(predict)), predict - predict_ste, predict + predict_ste, color='salmon', alpha=0.3)
plt.yticks(ticks = [0,0.2,0.4,0.6,0.8,1.0], labels=['Do', 'Re', 'Mi','Fa','Sol','La'], fontsize=12)  
plt.ylim([-0.4, 1.6])
for x in kk_tot:
    plt.axvline(x=x, color='k', linestyle=':', alpha=0.7)
    
fname_fig = os.path.join(fold_result, 'melody_correlation.png')
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
[label.set_fontsize(14) for label in ax.get_xticklabels()]
leg = plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
leg.get_frame().set_edgecolor('none')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.savefig(fname_fig, dpi= 300)

r, p = spearmanr(answers, predict)

#%%
'''
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(familiarity2, total_r2)
familiarity2_array = np.array(familiarity2)
regression_line = intercept + slope * familiarity2_array

plt.figure(figsize=(10,5))
plt.plot(familiarity2, total_r2, 'o', label='Original data', markersize=10)
plt.plot(familiarity2_array, regression_line, ':r', label=f'Regression line ($R^2$ = {r_value**2:.2f})')
plt.xlabel('Familiarity')
plt.ylabel('correlation coefficient of each song')
plt.title('Regression Line Plot')
plt.legend()
plt.show()
'''