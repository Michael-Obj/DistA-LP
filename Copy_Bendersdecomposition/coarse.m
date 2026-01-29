%% Header
addpath('./functions/');                                                    % Functions
addpath('./functions/myRLToolbox');                                         % My reinforcement learning toolbox
addpath('./functions/myBDToolbox');                                         % My Benders decomposition toolbox
addpath('./functions/myPlotToolbox');                                       % My plot toolbox
addpath('./classes');                                           % My plot toolbox
addpath('./functions/haversine');                                           % Read the Haversine distance package. This package is created by Created by Josiah Renfree, May 27, 2010
parameters;                                                                 % Read the parameters of the simulation

fprintf('------------------- Environment settings --------------------- \n \n'); 

%% Read the map information
fprintf("Loading the map information ... \n")
opts = detectImportOptions('./Dataset/rome/raw/rome_nodes.csv');
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable('./Dataset/rome/raw/rome_nodes.csv', opts);
df_edges = readtable('./Dataset/rome/raw/rome_edges.csv');

% opts = detectImportOptions('./Dataset/nyc/raw/nyc_nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./Dataset/nyc/raw/nyc_nodes.csv', opts);
% df_edges = readtable('./Dataset/nyc/raw/nyc_edges.csv');
% 
% opts = detectImportOptions('./Dataset/london/raw/london_nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./Dataset/london/raw/london_nodes.csv', opts);
% df_edges = readtable('./Dataset/london/raw/london_edges.csv');
                                                                            
col_longitude = table2array(df_nodes(:, 'x'));                              % Actual x (longitude) coordinate from the nodes data
col_latitude = table2array(df_nodes(:, 'y'));                               % Actual y (latitude) coordinate from the nodes data
col_osmid = table2array(df_nodes(:, 'osmid'));                              % Actual unique osmid from the nodes data
env_parameters.NR_LOC = size(col_longitude, 1); 

fprintf("The map information has been loaded. \n")
[G, u, v] = graph_preparation(df_nodes, df_edges);               % Given the map information, create the mobility graph
% load('u_london.mat');
% load('v_london.mat');
% load('G_london.mat'); %fast
% save('G_london.mat','G');
% save('u_london.mat','u');
% save('v_london.mat','v');
fprintf("The mobility graph has been created. \n \n")


max_longitude = max(col_longitude); 
min_longitude = min(col_longitude); 
LONGITUDE_SIZE = max_longitude - min_longitude; 

max_latitude = max(col_latitude);   
min_latitude = min(col_latitude);  
LATITUDE_SIZE = max_latitude - min_latitude; 


center_longitude = min_longitude + LONGITUDE_SIZE / 2;
center_latitude = min_latitude + LATITUDE_SIZE / 2;


target_longitude_half_size = LONGITUDE_SIZE / (2 * env_parameters.REGION_SCALE);
target_latitude_half_size = LATITUDE_SIZE / (2 * env_parameters.REGION_SCALE);


TARGET_LONGITUDE_MIN = center_longitude - target_longitude_half_size;
TARGET_LONGITUDE_MAX = center_longitude + target_longitude_half_size;
TARGET_LATITUDE_MIN = center_latitude - target_latitude_half_size;
TARGET_LATITUDE_MAX = center_latitude + target_latitude_half_size;


formatSpec = ['The target region is created: \n', ...
    'South-west corner coordinate: (latitude = %f, longitude = %f) \n', ...
    'North-east corner coordinate: (latitude = %f, longitude = %f) \n'];


fprintf(formatSpec, TARGET_LATITUDE_MIN, TARGET_LONGITUDE_MIN, TARGET_LATITUDE_MAX, TARGET_LONGITUDE_MAX);


xMin = min(col_longitude);
xMax = max(col_longitude);
yMin = min(col_latitude);
yMax = max(col_latitude);
num_grid=8;
edgesX = linspace(xMin, xMax, num_grid+1);
edgesY = linspace(yMin, yMax, num_grid+1);

repIndices = NaN(num_grid, num_grid);

for i = 1:num_grid
    for j = 1:num_grid

        if i < num_grid
            idxX = col_longitude >= edgesX(i) & col_longitude < edgesX(i+1);
        else
            idxX = col_longitude >= edgesX(i) & col_longitude <= edgesX(i+1);
        end
        
        if j < num_grid
            idxY = col_latitude >= edgesY(j) & col_latitude < edgesY(j+1);
        else
            idxY = col_latitude >= edgesY(j) & col_latitude <= edgesY(j+1);
        end
        
        idx = find(idxX & idxY);
        
        if ~isempty(idx)
            centerX = (edgesX(i) + edgesX(i+1)) / 2;
            centerY = (edgesY(j) + edgesY(j+1)) / 2;

            distances = sqrt((col_longitude(idx) - centerX).^2 + (col_latitude(idx) - centerY).^2);
            [~, minIdx] = min(distances);
            repIndices(i, j) = idx(minIdx);
        end
    end
end


pointIndicesArray = repIndices(~isnan(repIndices));
node_in_target_coarse = pointIndicesArray;
node_in_target_ori=node_in_target_coarse;
% env_parameters.NR_NODE_IN_TARGET=length(node_in_target_coarse);
% env_parameters.NR_OBFLOC=length(node_in_target_coarse);

distance_matrix = distanceMatrix(col_longitude(node_in_target_coarse), col_latitude(node_in_target_coarse));
distance_matrix_original=distance_matrix;

% path_distance_matrix = distances(mDPGraph);                                 % Calculate the path distance using the mDP graph






%% Parameters 
NR_RECORD = length(distance_matrix); 

EPSILON = env_parameters.EPSILON; 
DP_THRESHOLD = 2; 
NR_LOC=length(col_osmid);
node_in_target = randperm(NR_LOC, env_parameters.NR_NODE_IN_TARGET);


LPCA=[];
for EPSILON=1.5:0.5:15
    % node_in_target=load('node_in_target.mat');
    % node_in_target=node_in_target.node_in_target;
    
    obf_loc_coarse=randperm(size(node_in_target, 2), 100);
    % data_read = load('obf_loc_coarse.mat');
    % obf_loc_coarse=data_read.obf_loc_coarse;
    
    NR_OBF=length(obf_loc_coarse);
    %%
    Aeq = zeros(NR_RECORD, NR_RECORD*NR_OBF);
    
    for i = 1:NR_RECORD
        Aeq(i, (i-1)*NR_OBF+1:i*NR_OBF) = 1;
    end
    
    
    nPairs = NR_RECORD*(NR_RECORD-1)/2;
    nRows = 2 * NR_OBF * nPairs; 
    A = zeros(nRows, NR_RECORD*NR_OBF);
    b = zeros(nRows, 1);
    counter = 1;
    
    for i = 1:NR_RECORD
        for j = (i+1):NR_RECORD
            a_value = min(exp(EPSILON * distance_matrix(i,j)/10),1e8);
            for k = 1:NR_OBF
    
                A(counter, (i-1)*NR_OBF + k) = -1;
                A(counter, (j-1)*NR_OBF + k) = 1 / a_value;
                b(counter) = 0;
                counter = counter + 1;
                
                A(counter, (i-1)*NR_OBF + k) = 1;
                A(counter, (j-1)*NR_OBF + k) = -a_value;
                b(counter) = 0;
                counter = counter + 1;
            end
        end
    end
    
    tic;
    center_longitudes = col_longitude(pointIndicesArray);
    center_latitudes  = col_latitude(pointIndicesArray);
    target_longitudes = col_longitude(node_in_target);
    target_latitudes  = col_latitude(node_in_target);
    
    D = sqrt( (target_longitudes - center_longitudes').^2 + (target_latitudes - center_latitudes').^2 );
    
    [~, target_min] = min(D, [], 1);
    
    cost_50 = costMatrix_coarse(target_min, 2, obf_loc_coarse, G, 1:1:length(distance_matrix), node_in_target);
    c = reshape(cost_50', NR_RECORD*NR_OBF, 1);
    
    beq = ones(NR_RECORD, 1);
    lb = zeros(NR_RECORD*NR_OBF, 1);
    ub = ones(NR_RECORD*NR_OBF, 1);
    [x, fval] = linprog(c, A, b, Aeq, beq, lb, ub);
    loss_grid=fval;
    x_reshape=(reshape(x, NR_OBF, NR_RECORD))';
    
    
    
    
    
    
    
    
    center_longitudes = col_longitude(pointIndicesArray);
    center_latitudes  = col_latitude(pointIndicesArray);
    target_longitudes = col_longitude(node_in_target);
    target_latitudes  = col_latitude(node_in_target);
    
    D = sqrt( (target_longitudes - center_longitudes').^2 + (target_latitudes - center_latitudes').^2 );
    
    [~, target_grid] = min(D, [], 2);
    
    cost_500=costMatrix(node_in_target, 2, obf_loc_coarse, G, 1:1:env_parameters.NR_NODE_IN_TARGET);
    Perturbation_Matrix=x_reshape(target_grid,:);
    loss_coarse=sum(sum(cost_500.*Perturbation_Matrix))
    
    time_coarse=toc;
    
    %% violation ratio
    % R = 6371; 
    % lat_rad = deg2rad(target_latitudes);
    % lon_rad = deg2rad(target_longitudes);
    % dLat = lat_rad - lat_rad';  
    % dLon = lon_rad - lon_rad';  
    % a = sin(dLat/2).^2 + cos(lat_rad) .* cos(lat_rad)' .* sin(dLon/2).^2;
    % c = 2 .* atan2(sqrt(a), sqrt(1 - a));
    % distance1000 = R * c;

    % distance1000=distanceMatrix(target_longitudes, target_latitudes);
    % 
    % target_longitudes_hat = col_longitude(node_in_target_coarse(target_grid));
    % target_latitudes_hat  = col_latitude(node_in_target_coarse(target_grid));
    % distance_hathat=distanceMatrix(target_longitudes_hat, target_latitudes_hat);
    % vio_matrix=distance1000<distance_hathat;
    % vio_ratio=sum(vio_matrix(:))/(length(vio_matrix)*length(vio_matrix))



    % %%
    % task_loc=2;
    % NR_LOC = length(node_in_target); 
    % NR_OBFLOC = NR_OBF; 
    % cost_matrix = zeros(NR_LOC, NR_OBFLOC); 
    % for i = 1:1:NR_LOC
    %     [~, D] = shortestpathtree(G, node_in_target(task_loc)); % road
    %     %[~, D] = shortestpathtree(G, 1); % grid
    %     for j = 1:1:NR_OBFLOC         
    %         cost_matrix(i,j) = abs(D(node_in_target(i))-D(node_in_target(obf_loc_coarse(j)))); 
    %     end
    % end
    % cost = cost_matrix/NR_LOC; 
    % 
    % 
    % 
    % P_matrix=zeros(length(node_in_target),length(obf_loc_coarse));
    % sum_i=zeros(length(node_in_target),1);
    % epsilon=env_parameters.EPSILON;
    % for i=1:length(node_in_target)
    %     for j=1:length(obf_loc_coarse)
    %         [d, ~, ~] = haversine([col_longitude(node_in_target(i)), col_latitude(node_in_target(i))], [col_longitude(node_in_target(obf_loc_coarse(j))), col_latitude(node_in_target(obf_loc_coarse(j)))]);
    %         sum_i(i,1)=sum_i(i,1)+exp(-epsilon*d/2.0);
    %     end
    %     for j=1:length(obf_loc_coarse)
    %         [d, ~, ~] = haversine([col_longitude(node_in_target(i)), col_latitude(node_in_target(i))], [col_longitude(node_in_target(obf_loc_coarse(j))), col_latitude(node_in_target(obf_loc_coarse(j)))]);
    %         P_matrix(i,j)=exp(-epsilon*d/2.0)/sum_i(i,1);
    %     end
    % end
    % loss_exp = sum(sum(cost .* P_matrix))
    LPCA=[LPCA,loss_coarse];
end

figure;
plot(1.5:0.5:15, LPCA, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\epsilon Value', 'FontSize', 12);
ylabel('Utility Loss',   'FontSize', 12);
title('Utility Loss vs. Privacy Parameter \epsilon', 'FontSize', 14);
grid on;