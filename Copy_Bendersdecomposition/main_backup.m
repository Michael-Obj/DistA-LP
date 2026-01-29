addpath('./classes/Server/');
addpath('./classes/User/');
addpath('./classes/MasterProgram/');
addpath('./classes/Subproblem/');
addpath('./func/benchmarks/');
addpath('./func/benchmarks/randl/');
addpath('./func'); 
addpath('./func/read_files'); 
addpath('./func/haversine'); 

grid_size = 3; 
CRT_GRID_CELL_SIZE = 0.1; 

rng("default")

%% Parameters 
parameters; 


% ----------------------------
% ----------------------------
% ROME DATASET
% ----------------------------
% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1;
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;

% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.601; 
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.801;
% ----------------------------
% ----------------------------


% ----------------------------
% ----------------------------
% NYC DATASET
% ----------------------------
% TARGET_LON_MAX = -74; 
% TARGET_LON_MIN = -74.3; 
% TARGET_LAT_MAX = 40.65; 
% TARGET_LAT_MIN = 40.5;
 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.95; 
% TARGET_LAT_MIN = 40.801;

% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.8; 
% TARGET_LAT_MIN = 40.6501;

% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.65; 
% TARGET_LAT_MIN = 40.5;
% ----------------------------
% ----------------------------


% ----------------------------
% ----------------------------
% LONDON DATASET
% ----------------------------
% TARGET_LON_MAX = -0.3; 
% TARGET_LON_MIN = -0.5; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;

% TARGET_LON_MAX = -0.1; 
% TARGET_LON_MIN = -0.301; 
% TARGET_LAT_MAX = 51.7; 
% TARGET_LAT_MIN = 51.501;

% TARGET_LON_MAX = -0.1; 
% TARGET_LON_MIN = -0.301; 
% TARGET_LAT_MAX = 51.5; 
% TARGET_LAT_MIN = 51.3;

% TARGET_LON_MAX = 0.1; 
% TARGET_LON_MIN = -0.101; 
% TARGET_LAT_MAX = 51.7; 
% TARGET_LAT_MIN = 51.501;

% TARGET_LON_MAX = 0.1; 
% TARGET_LON_MIN = -0.101; 
% TARGET_LAT_MAX = 51.5; 
% TARGET_LAT_MIN = 51.3;

TARGET_LON_MAX = 0.3; 
TARGET_LON_MIN = 0.101; 
TARGET_LAT_MAX = 51.6; 
TARGET_LAT_MIN = 51.4;
% ----------------------------
% ----------------------------


env_parameters.longitude_min = TARGET_LON_MIN;
env_parameters.longitude_max = TARGET_LON_MAX; 
env_parameters.latitude_min = TARGET_LAT_MIN; 
env_parameters.latitude_max = TARGET_LAT_MAX; 

env_parameters.nr_loc_selected = 100; 
LR_LOC_SIZE = 20;                                                           % The total number of locations
OBF_RANGE = 20;                                                            % The obfuscation range is considered as a circle, and OBF_RANGE is the radius
EXP_RANGE = 10;                                                            % The set of location not applying exponential mechanism is within a circle, of which the radius is EXP_RANGE. 
NEIGHBOR_THRESHOLD = 0.5;                                                   % The neighbor threshold eta
NR_DEST = 1;                                                               % The number of destinations (spatial tasks)
NR_USER = 1;                                                                % The number of users (agents)

EPSILON = 2;                                                                % ??Michael
LR_SAMPLE_SIZE = 100;                                                       % ??Michael

NR_LOC = 1;
env_parameters.nr_loc_selected = NR_LOC*100; 


%% Initialization
env_parameters = readCityMapInfo(env_parameters);                         % Create the road map information of the target region: Rome, Italy
% env_parameters = readGridMapInfo(env_parameters);                           % Create the road map information of the target region: Rome, Italy
env_parameters.GAMMA = 1000; 
env_parameters.NEIGHBOR_THRESHOLD = 50;
    

%% Create the server
server = Server(NR_DEST, EXP_RANGE, CRT_GRID_CELL_SIZE);                    % Create the server
server = server.destination_identifier(env_parameters); 
server = server.cr_table_cal(env_parameters);                               % Create the cost reference table
% indist_set(grid_size, :) = threatByCostMatrix(server.cr_table, CRT_GRID_CELL_SIZE, 1); 
server.exp_range = EXP_RANGE; 


%% Create the users        
for m = 1:1:NR_USER
    user(m, 1) = User(m, LR_LOC_SIZE, OBF_RANGE, NEIGHBOR_THRESHOLD, env_parameters);               % Create users
    user(m, 1) = user(m, 1).initialization(env_parameters);                                         % Initialize the properties of the user, including the local relevant locations, distance matrices, obfuscated location IDs, and the cost matrix
end          
server = server.initialization(user);                                       % Create the destinations in the target region
        
for m = 1:1:NR_USER
    user(m, 1) = user(m, 1).cost_matrix_cal(server.cr_table, env_parameters);
end
 
       
%% Local relevant geo-obfuscation algorithm
tic;
server = server.geo_obfuscation_initialization(user, env_parameters);        
[server, user, nr_iterations, cost, cost_lower] = server.geo_obfuscation_generator(user, env_parameters);    % Generate the geo-obfuscation matrices 
computation_time = toc; 
% [nr_violations, violation_mag]= GeoInd_violation_cnt(user, env_parameters); 

save("cost.mat", "cost"); 
save("cost_lower.mat", "cost_lower"); 
save("nr_iterations.mat", "nr_iterations"); 
save("computation_time.mat", "computation_time"); 
