%% Header
addpath('./functions/');                                                    % Functions
addpath('./functions/myRLToolbox');                                         % My reinforcement learning toolbox
addpath('./functions/myBDToolbox');                                         % My Benders decomposition toolbox
addpath('./functions/myPlotToolbox');                                       % My plot toolbox

addpath('./functions/haversine');                                           % Read the Haversine distance package. This package is created by Created by Josiah Renfree, May 27, 2010
parameters;                                                                 % Read the parameters of the simulation
%env_parameters.NR_NODE_IN_TARGET=500;
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

% opts = detectImportOptions('./Dataset/london/raw/london_nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./Dataset/london/raw/london_nodes.csv', opts);
% df_edges = readtable('./Dataset/london/raw/london_edges.csv');
                                                                            
col_longitude = table2array(df_nodes(:, 'x'));                              % Actual x (longitude) coordinate from the nodes data
col_latitude = table2array(df_nodes(:, 'y'));                               % Actual y (latitude) coordinate from the nodes data
freq=table2array(df_nodes(:, 'street_count'));                          
env_parameters.NR_LOC = size(col_longitude, 1); 

fprintf("The map information has been loaded. \n")
[G, u, v] = graph_preparation(df_nodes, df_edges);               % Given the map information, create the mobility graph
% load('u_london.mat');
% load('v_london.mat');
% load('G_london.mat'); %fast

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




%% Find the set of nodes in the target region

NR_LOC=length(col_latitude);
node_in_target = randperm(NR_LOC, env_parameters.NR_NODE_IN_TARGET);
freq=freq(node_in_target);

%node_in_target = rand_sample(col_longitude,col_latitude,env_parameters);
%node_in_target = node_in_target_record{i_positionnn};
node_in_target_ori=node_in_target;


loc_x_in_target = col_longitude(node_in_target);                           
loc_y_in_target = col_latitude(node_in_target);
fprintf('The number of nodes is %d  \n', env_parameters.NR_NODE_IN_TARGET);

%% Perturbed locations are randomly distributed over the target region
obf_loc = randperm(size(node_in_target, 2), env_parameters.NR_OBFLOC);

env_parameters.obf_loc = obf_loc;
fprintf('The number of perturbed locations is %d  \n \n', env_parameters.NR_OBFLOC);


%% Distance matrix calculation                                                           
distance_matrix = distanceMatrix(col_longitude(node_in_target), col_latitude(node_in_target));
distance_matrix_original=distance_matrix;
adjacence_matrix = heaviside(1 - distance_matrix/env_parameters.NEIGHBOR_THRESHOLD);       % Create the adjacency matrix. 
adjacence_matrix_original=adjacence_matrix;
mDPMatrix = adjacence_matrix.*distance_matrix;                              % Create the mDP matrix. 
mDPGraph = graph(mDPMatrix);                                                % Create the mDP graph using the mDP matrix
% path_distance_matrix = distances(mDPGraph);                                 % Calculate the path distance using the mDP graph






%%
%num_user=5+floor(i_positionnn/10);
num_user=env_parameters.NR_AGENT;
user=randperm(env_parameters.NR_NODE_IN_TARGET, num_user);
k=20;

lambda =0.5;
alpha_hat=0.95;
delta=0.0000001;


D_MAX = max(max(distance_matrix));
D_MAX=50;
%range_threshold = D_MAX/(50*sqrt(env_parameters.NR_NODE_IN_TARGET/1000)*sqrt(env_parameters.NR_AGENT/50));
range_threshold = D_MAX/100;
threshold_matrix=distance_matrix<range_threshold;

omega=0.1;
w = getq(distance_matrix,lambda,range_threshold,alpha_hat);
[relevant_location_set, all_target] = get_relevant_location_set(w,user);
length(all_target)
epsilon_nmw = get_epsilon_nmw(w,all_target,distance_matrix,range_threshold);
B_xn_xnhat=get_B(w, distance_matrix, all_target);


%% get xi_nmhat
Pr=zeros(length(distance_matrix),length(distance_matrix));

for n_hat_ind=1:length(all_target)
    n_hat=all_target(n_hat_ind);
    for m_hat_ind=1:length(all_target)
        m_hat=all_target(m_hat_ind);
        if n_hat~=m_hat
            Pr(n_hat,m_hat)=0;
            % num_nearest_n=length(find(w(n_hat,:)>0.01));
            % [~, sortedIdx] = sort(distance_matrix(n_hat,:), 'ascend');
            % closest_nhat = sortedIdx(1:num_nearest_n);
            % num_nearest_m=length(find(w(m_hat,:)>0.01));


            num_nearest_n=length(find(distance_matrix(n_hat,:)<range_threshold));
            [~, sortedIdx] = sort(distance_matrix(n_hat,:), 'ascend');
            closest_nhat = sortedIdx(1:num_nearest_n);   
            num_nearest_m=length(find(distance_matrix(m_hat,:)<range_threshold));


            [~, sortedIdx] = sort(distance_matrix(m_hat,:), 'ascend');
            closest_mhat = sortedIdx(1:num_nearest_m);
            for n=1:length(closest_nhat)
                for m=1:length(closest_mhat)
                    sum_n=sum(w(:,n_hat));
                    sum_m=sum(w(:,m_hat));
                    Pr_nm=w(closest_nhat(n),n_hat)*w(closest_mhat(m),m_hat)/sum_n/sum_m;
                    Pr(n_hat,m_hat)=Pr(n_hat,m_hat)+Pr_nm;
                end
            end
        end
    end
end
[xi_hathat,xi_real]=get_xi_hathat(distance_matrix_original,epsilon_nmw,env_parameters.EPSILON,user,delta,w,Pr, all_target,threshold_matrix,B_xn_xnhat,range_threshold);

%%  actually mDP violation ratio
% num_vio=0;
% for n=1:length(user)
%     for m=1:length(user)
%         if ~ismember(user(n),relevant_location_set{user(n)})&&(n~=m)
%             n_hat=user(n);
%             m_hat=user(m);
%             if ~ismember(user(n),relevant_location_set{user(n)})
%                 [minValue_n, minIndex_n] = min(distance_matrix(user(n),relevant_location_set{user(n)}));
%                 n_hat=relevant_location_set{user(n)}(minIndex_n);
%             end
%             if ~ismember(user(m),relevant_location_set{user(m)})
%                 [minValue_m, minIndex_m] = min(distance_matrix(user(m),relevant_location_set{user(m)}));
%                 m_hat=relevant_location_set{user(m)}(minIndex_m);
%             end
%             if ((env_parameters.EPSILON-epsilon_nmw(n_hat,m_hat))*distance_matrix(n_hat,m_hat)-xi_hathat(find(all_target==n_hat),find(all_target==m_hat)))>(env_parameters.EPSILON-epsilon_nmw(user(n),user(m)))*distance_matrix(user(n),user(m))
%                 num_vio=num_vio+1;
%             end
%         end
%     end
% end
% violation_ratio=num_vio/(length(user)*(length(user)-1));
%%
[adjacence_matrix, distance_matrix, epsilon_nmw] = reget(adjacence_matrix, distance_matrix, all_target, epsilon_nmw);
env_parameters.NR_NODE_IN_TARGET=length(distance_matrix);
task_loc = 2;

freq=freq(1:length(all_target))/sum(freq(1:length(all_target)));
env_parameters.cost_matrix = costMatrix(node_in_target, task_loc, obf_loc, G, all_target, freq);             % Calculate the cost matrix
node_in_target_ori=node_in_target;
[loss_benchmarks,loss_Bayesian_Remapping,time_BR]=loss_for_benchmark(env_parameters, obf_loc, distance_matrix_original, node_in_target_ori, G, task_loc);
node_in_target = node_in_target(1,all_target);



privacy_budget=zeros(length(epsilon_nmw),length(epsilon_nmw));
for i_cout=1:length(distance_matrix)
    i=i_cout;
    for j_cout=1:length(distance_matrix)
        j=j_cout;
        if i~=j
            privacy_budget(i,j)=xi_hathat(i,j)/distance_matrix(i,j);
        end
    end
end
mean(privacy_budget(:))

%% 2PPO
% Cluster the nodes
env_parameters.NR_AGENT=25;
cluster_idx = kmeans(distance_matrix, env_parameters.NR_AGENT); 
% Create the agents
%env_parameters.NEIGHBOR_THRESHOLD=1;
fprintf('------------------- Create the agents ----------------------- \n'); 
tic;
agent_2PPO = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 

% Create the master agent
fprintf('------------------- Create the agents ----------------------- \n'); 
masteragent  = masterAgentCreation(distance_matrix, agent_2PPO, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
time2=toc;

% The algorithm starts here!!
tic;
ITER = 100; 
[~, ~, lowerbound, upperbound, upperbound_, loss1, obf_matrix] = bendersDecomposition(masteragent, agent_2PPO, env_parameters, ITER); 
time_2PPO=toc;
loss_matrix=env_parameters.cost_matrix.*obf_matrix;
loss=sum(loss_matrix(:));



%% LB
agent = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
masteragent  = masterAgentCreation(distance_matrix, agent, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
tic;
ITER = 100;
[~, ~, lowerbound_LB, upperbound_LB, upperbound__LB, loss_LB, obf_matrix_LB] = bendersDecomposition(masteragent, agent, env_parameters, ITER); 
time_LB=toc;
loss_matrix_LB=env_parameters.cost_matrix.*obf_matrix_LB;
loss_LB=sum(loss_matrix_LB(:));


% %% Benders
% parameters;
% %env_parameters.NEIGHBOR_THRESHOLD=1;
% env_parameters.cost_matrix = costMatrix(node_in_target_ori, task_loc, obf_loc, G, 1:length(node_in_target_ori));
% cluster_idx = kmeans(distance_matrix_original, env_parameters.NR_AGENT);
% zeros_matrix=sparse(length(distance_matrix_original),length(distance_matrix_original));
% tic;
% agent_benders = agentCreation(cluster_idx, node_in_target_ori, adjacence_matrix_original, distance_matrix_original, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, zeros_matrix, zeros_matrix); 
% fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 
% 
% % Create the master agent
% fprintf('------------------- Create the agents ----------------------- \n'); 
% 
% % Initialize the master agent
% masteragent  = masterAgentCreation(distance_matrix_original, agent_benders, adjacence_matrix_original, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, zeros_matrix, zeros_matrix); 
% time2_LB=toc;
% 
% 
% % The algorithm starts here!!
% tic;
% ITER = 100; 
% [~, ~, lowerbound_benders, upperbound_benders, upperbound__benders, loss_benders, obf_matrix_benders] = bendersDecomposition(masteragent, agent_benders, env_parameters, ITER); 
% 
% time_benders=toc;
% 
% loss_matrix_benders=env_parameters.cost_matrix.*obf_matrix_benders;
% loss_benders=sum(loss_matrix_benders(:));


% xi_d_d = zeros(size(distance_matrix));
% mask = (xi_hathat ~= 0);
% xi_d_d(mask) = distance_matrix(mask) ./ xi_hathat(mask);

% 
% non_zero_values = xi_hathat(xi_hathat ~= 0);
% non_zero_values = xi_real(xi_real ~= 0);
% non_zero_values = distance_matrix(distance_matrix ~= 0);
% non_zero_values = B_xn_xnhat(B_xn_xnhat >0);
% non_zero_values = xi_d_d(xi_d_d >0);
% 
% mean_value = mean(non_zero_values);
% 
% 
% variance_value = var(non_zero_values);
% 
% 
% std_value = std(non_zero_values);
% 
% 
% max_value = max(non_zero_values);
% 
% 
% min_value = min(non_zero_values);
% 
% 
% fprintf('mean: %.2f\n', mean_value);
% fprintf('variance: %.2f\n', variance_value);
% fprintf('std: %.2f\n', std_value);
% fprintf('max_value: %.2f\n', max_value);
% fprintf('min_value: %.2f\n', min_value);


%time_2PPO
ep=min(env_parameters.EPSILON,epsilon_nmw);

time_2PPO
mean(ep(:))
mean(privacy_budget(:))







loss_epsilon_1=[];
EM=[];
EMBR=[];
time=[];
for epsilon_value=1.5:0.5:15
    env_parameters.EPSILON=epsilon_value;
    % Create the agents
    %env_parameters.NEIGHBOR_THRESHOLD=1;
    fprintf('------------------- Create the agents ----------------------- \n'); 
    tic;
    agent_2PPO = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
    fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 

    % Create the master agent
    fprintf('------------------- Create the agents ----------------------- \n'); 
    masteragent  = masterAgentCreation(distance_matrix, agent_2PPO, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
    time2=toc;

    % The algorithm starts here!!
    tic;
    ITER = 100; 
    [~, ~, lowerbound, upperbound, upperbound_, loss1, obf_matrix] = bendersDecomposition(masteragent, agent_2PPO, env_parameters, ITER); 
    time_2PPO=toc;
    loss_matrix=env_parameters.cost_matrix.*obf_matrix;
    loss=sum(loss_matrix(:));
    loss_epsilon_1=[loss_epsilon_1,loss];
    [loss_benchmarks,loss_Bayesian_Remapping,time_BR]=loss_for_benchmark(env_parameters, obf_loc, distance_matrix_original, node_in_target_ori, G, task_loc);
    EM=[EM,loss_benchmarks];
    EMBR=[EMBR,loss_Bayesian_Remapping];
    time=[time,time_2PPO];
end

% 
% figure;
% plot(1.5:0.5:15, time, '-o', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('\epsilon Value', 'FontSize', 12);
% ylabel('Utility Loss',   'FontSize', 12);
% title('Utility Loss vs. Privacy Parameter \epsilon', 'FontSize', 14);
% grid on;
% 
% 
% 
% figure;
% plot(1.5:0.5:15, EM, '-o', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('\epsilon Value', 'FontSize', 12);
% ylabel('Utility Loss',   'FontSize', 12);
% title('Utility Loss vs. Privacy Parameter \epsilon', 'FontSize', 14);
% grid on;
% 
% 
% figure;
% plot(1.5:0.5:15, EMBR, '-o', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('\epsilon Value', 'FontSize', 12);
% ylabel('Utility Loss',   'FontSize', 12);
% title('Utility Loss vs. Privacy Parameter \epsilon', 'FontSize', 14);
% grid on;