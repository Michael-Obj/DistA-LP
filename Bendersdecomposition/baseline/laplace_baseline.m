for user_data = 1:10
    loss_baseline=zeros(9,4);  % change
    for r_data = 1:4                % change
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
        distance_matrix = distanceMatrix(col_longitude(node_tar), col_latitude(node_tar));
        exp_4=exp(4*distance_matrix);
        exp_7=exp(7*distance_matrix);
        exp_10=exp(10*distance_matrix);
        %% cost
        NR_LOC = length(node_tar); 
        NR_OBFLOC = size(obf_ID, 2); 
        cost_matrix = zeros(NR_LOC, NR_OBFLOC); 
        for i = 1:1:NR_LOC
            [~, D] = shortestpathtree(G, node_tar(2)); 
            for j = 1:1:NR_OBFLOC         
                cost_matrix(i,j) = abs(D(node_tar(i))-D(node_tar(obf_ID(j)))); 
            end
        end
        cost_matrix = cost_matrix/NR_LOC;
        %%
        loc_lons=lon_sel;
        loc_lats = lat_sel; 
        pert_lons = col_longitude(obf_ID');
        pert_lats = col_latitude(obf_ID');
        % epsilon4
        [K, QL, time_laplace4] = planar_laplace_utility_loss( ...
        loc_lons, loc_lats, ...
        pert_lons, pert_lats, ...
        cost_matrix, 4);
        loss_4=sum(sum(cost_matrix .* K));
        violation_matrix4=zeros(length(node_tar),length(node_tar));
        % for i=1:length(node_tar)
        %     for j=1:length(node_tar)
        %         for k=1:length(obf_ID)
        %             if K(i,k)/K(j,k)>exp_4(i,j)
        %                 violation_matrix4(i,j)=violation_matrix4(i,j)+1;
        %             end
        %         end
        %     end
        % end
        violation_rate4=sum(violation_matrix4(:))/(length(node_tar)*length(node_tar)*length(obf_ID));
        % epsilon7
        [K, QL, time_laplace7] = planar_laplace_utility_loss( ...
        loc_lons, loc_lats, ...
        pert_lons, pert_lats, ...
        cost_matrix, 7);
        loss_7=sum(sum(cost_matrix .* K));
        violation_matrix7=zeros(length(node_tar),length(node_tar));
        % for i=1:length(node_tar)
        %     for j=1:length(node_tar)
        %         for k=1:length(obf_ID)
        %             if K(i,k)/K(j,k)>exp_7(i,j)
        %                 violation_matrix7(i,j)=violation_matrix7(i,j)+1;
        %             end
        %         end
        %     end
        % end
        violation_rate7=sum(violation_matrix7(:))/(length(node_tar)*length(node_tar)*length(obf_ID));
        % epsilon10
        [K, QL, time_laplace10] = planar_laplace_utility_loss( ...
        loc_lons, loc_lats, ...
        pert_lons, pert_lats, ...
        cost_matrix, 10);
        loss_10=sum(sum(cost_matrix .* K));
        violation_matrix10=zeros(length(node_tar),length(node_tar));
        % for i=1:length(node_tar)
        %     for j=1:length(node_tar)
        %         for k=1:length(obf_ID)
        %             if K(i,k)/K(j,k)>exp_10(i,j)
        %                 violation_matrix10(i,j)=violation_matrix10(i,j)+1;
        %             end
        %         end
        %     end
        % end
        violation_rate10=sum(violation_matrix10(:))/(length(node_tar)*length(node_tar)*length(obf_ID));
        % loss
        loss_baseline(1,r_data)=loss_4;
        loss_baseline(2,r_data)=time_laplace4;
        loss_baseline(3,r_data)=violation_rate4;

        loss_baseline(4,r_data)=loss_7;
        loss_baseline(5,r_data)=time_laplace7;
        loss_baseline(6,r_data)=violation_rate7;

        loss_baseline(7,r_data)=loss_10;
        loss_baseline(8,r_data)=time_laplace10;
        loss_baseline(9,r_data)=violation_rate10;
    end
    save_name=sprintf('loss_baseline_nyc_user%d.mat', user_data);
    save(save_name,'loss_baseline');
    clear;
end


%%
dataDir = fullfile('.', 'baseline_laplace', 'nyc', '6000');

% File name pattern
filePattern = fullfile(dataDir, 'loss_baseline_nyc_user*.mat');

% Get the list of files
files = dir(filePattern);

% Initialize the accumulator matrix
sumMatrix = zeros(9,4);              %change

% Iterate over the files and accumulate the results
for k = 1:length(files)
    filePath = fullfile(dataDir, files(k).name);
    data = load(filePath);      
    sumMatrix = sumMatrix + data.loss_baseline;  
end

% Display the result
disp('Final accumulated 9-by-6 matrix:');
disp(sumMatrix);
sumMatrix=sumMatrix(1:9,1:4);            % change
save(fullfile(dataDir, 'sumMatrix_nyc.mat'), 'sumMatrix');

sumMatrix(2:3,:)=sumMatrix(2:3,:)/10;
sumMatrix(5:6,:)=sumMatrix(5:6,:)/10;
sumMatrix(8:9,:)=sumMatrix(8:9,:)/10;
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
for i = 1:1
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 2:3
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end

for i = 4:4
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 5:6
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end

for i = 7:7
    fprintf('Row %d: %.2f ± %.2f\n', i, row_mean(i)/10000, row_std(i)*0.6198/10000);
end

for i = 8:9
    fprintf('Row %d: %.4f ± %.4f\n', i, row_mean(i), row_std(i));
end
