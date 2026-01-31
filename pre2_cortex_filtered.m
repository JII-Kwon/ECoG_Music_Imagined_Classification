clearvars -except SubName Sub_Nums sub ...
    Bad_Ch Ch_list NumCh main Info SubNum SubToken

fold_mat = [ main '\01_mat_file' ];
%% Parameter
band_names = [ "D" "T" "A" "B" "G" "HG"];
Bands = [ 1 4 4 8 8 12 12 30 30 59 61 150];

%% Filtering
dirs = dir([fold_mat,'\' SubToken '_session_*.mat']);
mat_names={};
for ii=1:length(dirs)
    mat_names{ii} = dirs(ii).name;
end
for bd = 1 : length(band_names)
    nbd = convertStringsToChars(band_names(bd));
    name_band = [num2str(bd) '_' nbd];
    frq_ini = Bands(2*bd-1);
    frq_fin = Bands(2*bd);
    
    fold_filtered = [fold_mat '\' name_band ];
    
    if exist(fold_filtered, 'dir') == 0, mkdir(fold_filtered); end
    
    clearvars temp
    
    
    for ic = 1 : length(mat_names)
        fn_mat = mat_names(ic); fn_mat = fn_mat{1,1};
        

        name_filtered = [fold_filtered '\' name_band '_' fn_mat];
        
        name_ecog =  [ fold_mat '\' fn_mat];
    
        if exist(name_ecog, 'file') == 0
           continue
        end
        ecog = load(name_ecog);
        
        srate = ecog.srate;
        NumCh = ecog.NumCh;
        Bad_Ch = ecog.Bad_Ch;
        Ch_list = ecog.Ch_list;
        trig = ecog.event;
                
        save(name_filtered, 'SubName', 'NumCh', 'trig', 'srate', 'Bad_Ch', 'Ch_list')
        
        data = ecog.data;
        
        fdata = eegfilt(data, srate, frq_ini, frq_fin);
        
        save(name_filtered,  'fdata', '-append')
        
        clearvars data srate NumCh Bad_Ch Ch_list trig
        
    end
    
end
fprintf([SubName '...Filtering is done... \n'])

