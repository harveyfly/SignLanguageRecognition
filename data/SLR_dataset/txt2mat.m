clc, clear;
data_dir = '.\xf500_body_depth_txt';
save_dir = '.\xf500_body_depth_mat';
data_num = 500;
for index = 0 : data_num - 1
   index
   label_name = num2str(index, '%06d');
   one_label_data_path = strcat(data_dir, '\', label_name);
   txt_files = {dir(fullfile(one_label_data_path, '*.txt')).name};
   mkdir(save_dir, label_name);
   for i = 1 : length(txt_files)
       file_name = strcat(one_label_data_path, '\', txt_files(i));
       fp = fopen(string(file_name), 'r');
       % ∂¡»°txt
       C = cell2mat(textscan(fp, '%.1f'));
       data = reshape(C, 50, [])';
       save_data_path = strcat(save_dir, '\', label_name, '\', num2str(i), '.mat');
       save(save_data_path, 'data');
%        txt_data = txt_data(round(linspace(1, size(txt_data, 1), 50)), :);
%        data = [data, txt_data];
       fclose(fp);
   end
end
