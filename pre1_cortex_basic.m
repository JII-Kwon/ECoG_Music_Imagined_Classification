clearvars -except SubName Sub_Nums sub ...
    Bad_Ch Ch_list NumCh main Info SubNum SubToken

fold_raw = [main '\00_raw_data\', SubToken];
fold_mat = [ main '\01_mat_file' ];
fold_fig = [fold_mat '\01_figure_aft_pcs\' SubName ];
if exist(fold_fig, 'dir') == 0, mkdir(fold_fig); end

dirs = dir([fold_raw,'\session_*.cnt']); 
cnt_names={};
for ii=1:length(dirs)
    cnt_names{ii} = dirs(ii).name;
end

for ic = 1 : length(cnt_names)
    fn_cnt = cnt_names(ic); fn_cnt = fn_cnt{1,1};
    
    cnt = loadcnt([fold_raw '\'  fn_cnt ]);
    
    data = cnt.data(1:NumCh,:);
    event = cnt.event;
    srate = cnt.header.rate;

    data(Bad_Ch, :) = [];
    
    if convertCharsToStrings(class(data)) == "single"
        data = double(data);
    end
   
    clearvars cnt
    data_or = data;
    data = eegfilt(data,srate,0.1,0);
         
    CARdata = mean(data,1);
    for i = 1: size(data,1)
        data(i,:) = data(i,:) - CARdata;
    end

    for ch = 1 : size(data,1)
        
        fig = figure;
        set(fig,'Visible', 'off');
        Ch_num = Ch_list(ch);
        name_fig = [ fold_fig '\' SubName '_Ch' num2str(Ch_num) '_sess' num2str(ic)];
        subplot(2,1,1), plot(data(ch,:))
        title([ 'Ch' num2str(Ch_num) ])
        
        subplot(2,1,2), plot(data_or(ch,:))
        print('-dpng', name_fig)
        close all
    end
    
    
    for not = 1 : floor(1000/60)
        data = NotchFilter_ryun(data, srate, 60*not);
    end
    
    fprintf('.');
    
    
    file_name = [ fold_mat '\' SubToken '_session_' num2str(ic) '.mat'];
    
    save(file_name, 'data', 'event', 'NumCh', 'srate', 'Bad_Ch', 'Ch_list')

end