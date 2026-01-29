%% Read the map information
addpath('./functions/');                                                    % Functions

fprintf("Loading the map information ... \n")
opts = detectImportOptions('./datasets/rome/rome_nodes.csv');
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable('./datasets/rome/rome_nodes.csv', opts);
df_edges = readtable('./datasets/rome/rome_edges.csv');
                                                                            
col_longitude = table2array(df_nodes(:, 'x'));                              % Actual x (longitude) coordinate from the nodes data
col_latitude = table2array(df_nodes(:, 'y'));                               % Actual y (latitude) coordinate from the nodes data
col_osmid = table2array(df_nodes(:, 'osmid'));                              % Actual unique osmid from the nodes data
env_parameters.NR_LOC = size(col_longitude, 1); 

fprintf("The map information has been loaded. \n")
[G, u, v] = graph_preparation(df_nodes, df_edges);               % Given the map information, create the mobility graph
% load('u_london.mat');
% load('v_london.mat');
% load('G_london.mat'); %fast

fprintf("The mobility graph has been created. \n \n")


%% This is the just an example of using the function "node_in_target" to find the locations within the target region
TARGET_LON_MAX = 12.5647; 
TARGET_LON_MIN = 12.5361; 
TARGET_LAT_MAX = 41.8883; 
TARGET_LAT_MIN = 41.8662;

[col_osmid_selected, col_longitude_selected, col_latitude_selected] = node_in_target(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN); 

% The following plot is just for testing 
plot(col_longitude, col_latitude, 'o'); 
hold on; 
plot(col_longitude_selected, col_latitude_selected, 'o'); 
