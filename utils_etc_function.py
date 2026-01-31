from utils_basic_music import np, pd, plt, sns
from collections import Counter

class Utils:
    def concate_major(self, file, sd, Test_sess):
        data1 = file['norm_C_'+sd]
        data2 = file['norm_D_'+sd]
        data3 = file['norm_B_'+sd]
        data = np.concatenate((data1[:, :, :int((Test_sess-1)*data1.shape[2]/Test_sess)], data2[:, :, :int((Test_sess-1)*data2.shape[2]/Test_sess)]), axis =2)
        data = np.concatenate((data, data3[:, :, :int((Test_sess-1)*data3.shape[2]/Test_sess)]), axis =2)
        
        data_test = np.concatenate((data1[:, :, int((Test_sess-1)*data1.shape[2]/Test_sess):], data2[:, :, int((Test_sess-1)*data2.shape[2]/Test_sess):]), axis =2)
        data_test = np.concatenate((data_test, data3[:, :, int((Test_sess-1)*data3.shape[2]/Test_sess):]), axis =2)
            
        return data, data_test
  
    
    def feature_extract(self, data, select_elect, mean_max ='mean'):
        data2 = data[select_elect,:,:]
        if mean_max == 'mean':
            data2 = np.mean(data2, axis = 1)
        elif mean_max == 'max':
            data2 = np.max(data2, axis = 1)
        
        return data2
    
    def feature_extract_sess(self, sd, select_elect, mean_max ='mean'):
        data2 = data[select_elect,:,:]
        if mean_max == 'mean':
            data2 = np.mean(data2, axis = 1)
        elif mean_max == 'max':
            data2 = np.max(data2, axis = 1)
        
        return data2
    
    
    def mk_multidx(self, col0, col1, index0, index1=None, data=np.nan):
        columns = pd.MultiIndex.from_product([col0, col1])
        if index1 == None:
            index = index0
        else:
            index = pd.MultiIndex.from_product([index0,index1])
        df = pd.DataFrame(data=data, index=index, columns=columns)
        return df
    
        
    def create_df(self, dataframes_dict, val_name = 'feature_sd', range_sd = np.arange(1,7), axis=0):
        
        dfs_to_concatenate = [dataframes_dict[val_name + str(sd)] for sd in range_sd]
        df = pd.concat(dfs_to_concatenate, axis=axis, ignore_index=(axis == 0))
                
        return df
    
    
    
    