for user_data = 1:10
    loss_baseline=zeros(12,6);  % change
    for r_data = 1:6                % change
        filename = sprintf('./nyc_location_data_6000_nodes/location_data_sample_%d/location_data_r%d_user%d.mat', user_data, r_data, user_data);
        load(filename);
        addpath('./functions/');
        addpath('./functions/haversine');
        opts = detectImportOptions('./Dataset/nyc/raw/nyc_nodes.csv');
        opts = setvartype(opts, 'osmid', 'int64');
        df_nodes = readtable('./Dataset/nyc/raw/nyc_nodes.csv', opts);
        df_edges = readtable('./Dataset/nyc/raw/nyc_edges.csv');
        col_longitude = table2array(df_nodes(:, 'x'));
        col_latitude = table2array(df_nodes(:, 'y'));
        parameters;
        % [G, u, v] = graph_preparation(df_nodes, df_edges);
        load('u_nyc.mat');
        load('v_nyc.mat');
        load('G_nyc.mat');

        % PAnDA;
        PAnDA_memory_instrumented;
        % loss
        loss_baseline(1,r_data)=loss_ep4;
        loss_baseline(2,r_data)=loss_LB_ep4;
        loss_baseline(3,r_data)=time_2PPO_ep4;
        loss_baseline(4,r_data)=time_LB_ep4;

        loss_baseline(5,r_data)=loss_ep7;
        loss_baseline(6,r_data)=loss_LB_ep7;
        loss_baseline(7,r_data)=time_2PPO_ep7;
        loss_baseline(8,r_data)=time_LB_ep7;

        loss_baseline(9,r_data)=loss_ep10;
        loss_baseline(10,r_data)=loss_LB_ep10;
        loss_baseline(11,r_data)=time_2PPO_ep10;
        loss_baseline(12,r_data)=time_LB_ep10;
    end
    save_name=sprintf('loss_baseline_nyc_user%d.mat', user_data);
    save(save_name,'loss_baseline');
    clear;
end


%%
dataDir = fullfile('.', 'baseline_PAnDA', 'nyc', '6000');

% File name pattern
filePattern = fullfile(dataDir, 'loss_baseline_nyc_user*.mat');

% Get the list of files
files = dir(filePattern);

% Initialize the accumulator matrix
sumMatrix = zeros(12,6);              %change

% Iterate over the files and accumulate the results
for k = 1:length(files)
    filePath = fullfile(dataDir, files(k).name);
    data = load(filePath);      
    sumMatrix = sumMatrix + data.loss_baseline;  
end

% Display the result
disp('Final accumulated 12-by-6 matrix:');
disp(sumMatrix);
sumMatrix=sumMatrix(1:12,1:6);            % change
save(fullfile(dataDir, 'sumMatrix_nyc.mat'), 'sumMatrix');

sumMatrix(3:4,:)=sumMatrix(3:4,:)/10;
sumMatrix(7:8,:)=sumMatrix(7:8,:)/10;
sumMatrix(11:12,:)=sumMatrix(11:12,:)/10;
%% mean std
% Assume the matrix is named sumMatrix

% Mean of each row
row_mean = mean(sumMatrix, 2);

% Standard deviation of each row
row_std = std(sumMatrix, 0, 2);

% Display the results as "mean +/- standard deviation"
% disp('Mean +/- standard deviation for each row:');
% for i = 1:3
%     fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)/10000);
% end
disp('Mean +/- standard deviation for each row:');
for i = 1:2
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 3:4
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end

for i = 5:6
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 7:8
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end

for i = 9:10
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 11:12
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end
