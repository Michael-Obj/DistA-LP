% load('location_data_r5_user2.mat');
% addpath('./functions/'); 
% addpath('./functions/haversine');
% opts = detectImportOptions('./Dataset/london/raw/london_nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./Dataset/london/raw/london_nodes.csv', opts);
% df_edges = readtable('./Dataset/london/raw/london_edges.csv');
% col_longitude = table2array(df_nodes(:, 'x'));  
% col_latitude = table2array(df_nodes(:, 'y'));
% parameters;
% [G, u, v] = graph_preparation(df_nodes, df_edges);
% distance_matrix = distanceMatrix(col_longitude(node_tar), col_latitude(node_tar));
% task_loc=2;
% [loss_benchmarks,loss_Bayesian_Remapping,~]=loss_for_benchmark(env_parameters, obf_ID, distance_matrix, node_tar, G, task_loc)
% 
% 
% 
% coarse;
% %% compare with randomly selected locations
% NR_LOC=length(col_latitude);
% node_tar_random = randperm(NR_LOC, env_parameters.NR_NODE_IN_TARGET);
% obf_ID_random = randperm(size(node_tar_random, 2), length(obf_ID));
% distance_matrix = distanceMatrix(col_longitude(node_tar_random), col_latitude(node_tar_random));
% [loss_benchmarks2,loss_Bayesian_Remapping2,~]=loss_for_benchmark(env_parameters, obf_ID_random, distance_matrix, node_tar_random, G, task_loc)


%% EM, EMBR, LPCA
% 
for user = 1:10
    loss_baseline=zeros(7,6);  % change
    for r = 1:6                % change
        filename = sprintf('./london_location_data_2000_nodes/location_data_sample_%d/location_data_r%d_user%d.mat', user, r, user);
        load(filename);
        addpath('./functions/');
        addpath('./functions/haversine');
        opts = detectImportOptions('./Dataset/london/raw/london_nodes.csv');
        opts = setvartype(opts, 'osmid', 'int64');
        df_nodes = readtable('./Dataset/london/raw/london_nodes.csv', opts);
        df_edges = readtable('./Dataset/london/raw/london_edges.csv');
        col_longitude = table2array(df_nodes(:, 'x'));
        col_latitude = table2array(df_nodes(:, 'y'));
        parameters;
        % [G, u, v] = graph_preparation(df_nodes, df_edges);
        load('u_london.mat');
        load('v_london.mat');
        load('G_london.mat');
        distance_matrix = distanceMatrix(col_longitude(node_tar), col_latitude(node_tar));
        distance_save=distance_matrix;
        task_loc=2;
        [loss_benchmarks,loss_Bayesian_Remapping,~,time_EM,time_BR]=loss_for_benchmark(env_parameters, obf_ID, distance_matrix, node_tar, G, task_loc)
        coarse;
        % loss
        loss_baseline(1,r)=loss_benchmarks;
        loss_baseline(2,r)=loss_Bayesian_Remapping;
        loss_baseline(3,r)=loss_coarse;
        % time
        loss_baseline(4,r)=time_EM;
        loss_baseline(5,r)=time_BR;
        loss_baseline(6,r)=time_LPCA;
        %vio_ratio
        loss_baseline(7,r)=vio_ratio;
    end
    save_name=sprintf('loss_baseline_london_user%d.mat', user);
    save(save_name,'loss_baseline');
    clear;
end

%% COPT