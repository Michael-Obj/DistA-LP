%% Header
addpath('./functions/');                                                    % Functions
addpath('./functions/myRLToolbox');                                         % My reinforcement learning toolbox
addpath('./functions/myBDToolbox');                                         % My Benders decomposition toolbox
addpath('./functions/myPlotToolbox');                                       % My plot toolbox
addpath('./functions/haversine');                                           % Read the Haversine distance package. This package is created by Created by Josiah Renfree, May 27, 2010

panda_mem_output_dir = fullfile(pwd, 'memory_cost_probe_outputs');
if ~exist(panda_mem_output_dir, 'dir')
    mkdir(panda_mem_output_dir);
end
panda_mem_tag = getenv('PANDA_MEM_TAG');
if strlength(string(panda_mem_tag)) == 0
    panda_mem_tag = datestr(now, 'yyyymmdd_HHMMSS');
end
panda_mem_csv = fullfile(panda_mem_output_dir, sprintf('PAnDA_memory_%s_live.csv', panda_mem_tag));
panda_mem_mat = fullfile(panda_mem_output_dir, sprintf('PAnDA_memory_%s_live.mat', panda_mem_tag));
panda_mem_records = table();
panda_mem_t0 = tic;
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'start', panda_mem_t0, ...
    {'node_tar','obf_ID','env_parameters'}, panda_mem_csv, panda_mem_mat);

%fprintf('------------------- Environment settings --------------------- \n \n'); 

%% Read the map information

freq=table2array(df_nodes(:, 'street_count'));                          
env_parameters.NR_LOC = size(col_longitude, 1); 



%% Find the set of nodes in the target region

NR_LOC=length(col_latitude);
node_in_target = node_tar;
freq=freq(node_in_target);
node_in_target_ori=node_in_target;


loc_x_in_target = col_longitude(node_in_target);                           
loc_y_in_target = col_latitude(node_in_target);
%fprintf('The number of nodes is %d  \n', env_parameters.NR_NODE_IN_TARGET);

%% Perturbed locations are randomly distributed over the target region
obf_loc = obf_ID;

env_parameters.obf_loc = obf_loc;
%fprintf('The number of perturbed locations is %d  \n \n', env_parameters.NR_OBFLOC);


%% Distance matrix calculation                                                           
distance_matrix = distanceMatrix(col_longitude(node_in_target), col_latitude(node_in_target));
distance_matrix_original=distance_matrix;
adjacence_matrix = heaviside(1 - distance_matrix/env_parameters.NEIGHBOR_THRESHOLD);       % Create the adjacency matrix. 
adjacence_matrix_original=adjacence_matrix;
mDPMatrix = adjacence_matrix.*distance_matrix;                              % Create the mDP matrix. 
mDPGraph = graph(mDPMatrix);                                                % Create the mDP graph using the mDP matrix
% path_distance_matrix = distances(mDPGraph);                                 % Calculate the path distance using the mDP graph
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_distance_matrix', panda_mem_t0, ...
    {'distance_matrix','distance_matrix_original','adjacence_matrix','mDPMatrix','mDPGraph'}, panda_mem_csv, panda_mem_mat);






%%
%num_user=5+floor(i_positionnn/10);
num_user=env_parameters.NR_AGENT;
user=randperm(env_parameters.NR_NODE_IN_TARGET, num_user);

lambda =0.5;
alpha_hat=0.95;
delta=0.0000001;


D_MAX = max(max(distance_matrix));

%range_threshold = D_MAX/(50*sqrt(env_parameters.NR_NODE_IN_TARGET/1000)*sqrt(env_parameters.NR_AGENT/50));
range_threshold = D_MAX/150;
threshold_matrix=distance_matrix<range_threshold;

omega=0.1;
w = getq(distance_matrix,lambda,range_threshold,alpha_hat);
[relevant_location_set, all_target] = get_relevant_location_set(w,user);
length(all_target);
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_selected_set', panda_mem_t0, ...
    {'w','relevant_location_set','all_target','distance_matrix','threshold_matrix'}, panda_mem_csv, panda_mem_mat);
epsilon_nmw = get_epsilon_nmw(w,all_target,distance_matrix,range_threshold);
B_xn_xnhat=get_B(w, distance_matrix, all_target);
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_phase1_params', panda_mem_t0, ...
    {'epsilon_nmw','B_xn_xnhat','w','all_target'}, panda_mem_csv, panda_mem_mat);


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
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_xi_hathat', panda_mem_t0, ...
    {'Pr','xi_hathat','xi_real','epsilon_nmw','B_xn_xnhat'}, panda_mem_csv, panda_mem_mat);

%%
[adjacence_matrix, distance_matrix, epsilon_nmw] = reget(adjacence_matrix, distance_matrix, all_target, epsilon_nmw);
env_parameters.NR_NODE_IN_TARGET=length(distance_matrix);
task_loc = 2;
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_reduced_problem', panda_mem_t0, ...
    {'adjacence_matrix','distance_matrix','epsilon_nmw','all_target'}, panda_mem_csv, panda_mem_mat);

freq=freq(1:length(all_target))/sum(freq(1:length(all_target)));
env_parameters.cost_matrix = costMatrix(node_in_target, task_loc, obf_loc, G, all_target, freq);             % Calculate the cost matrix
node_in_target_ori=node_in_target;
[loss_benchmarks,loss_Bayesian_Remapping,time_BR]=loss_for_benchmark(env_parameters, obf_loc, distance_matrix_original, node_in_target_ori, G, task_loc);
node_in_target = node_in_target(1,all_target);
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_cost_and_benchmarks', panda_mem_t0, ...
    {'env_parameters','loss_benchmarks','loss_Bayesian_Remapping','node_in_target','node_in_target_ori'}, panda_mem_csv, panda_mem_mat);



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
mean(privacy_budget(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'after_privacy_budget', panda_mem_t0, ...
    {'privacy_budget','epsilon_nmw','xi_hathat'}, panda_mem_csv, panda_mem_mat);

%% 2PPO epsilon=4
% Cluster the nodes
env_parameters.EPSILON=4;
env_parameters.NR_AGENT=25;
cluster_idx = kmeans(distance_matrix, env_parameters.NR_AGENT); 
% Create the agents
%env_parameters.NEIGHBOR_THRESHOLD=1;
%fprintf('------------------- Create the agents ----------------------- \n'); 
tic;
agent_2PPO = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
%fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 

% Create the master agent
%fprintf('------------------- Create the agents ----------------------- \n'); 
masteragent  = masterAgentCreation(distance_matrix, agent_2PPO, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
time2=toc;
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps4_after_agent_master_2ppo', panda_mem_t0, ...
    {'cluster_idx','agent_2PPO','masteragent'}, panda_mem_csv, panda_mem_mat);

% The algorithm starts here!!
tic;
ITER = 100; 
[~, ~, lowerbound, upperbound, upperbound_, loss1, obf_matrix] = bendersDecomposition(masteragent, agent_2PPO, env_parameters, ITER); 
time_2PPO_ep4=toc;
loss_matrix=env_parameters.cost_matrix.*obf_matrix;
loss_ep4=sum(loss_matrix(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps4_after_benders_2ppo', panda_mem_t0, ...
    {'agent_2PPO','masteragent','lowerbound','upperbound','upperbound_','obf_matrix','loss_matrix'}, panda_mem_csv, panda_mem_mat);



%% LB epsilon=4
env_parameters.EPSILON=4;
agent = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
masteragent  = masterAgentCreation(distance_matrix, agent, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps4_after_agent_master_lb', panda_mem_t0, ...
    {'agent','masteragent'}, panda_mem_csv, panda_mem_mat);
tic;
ITER = 100;
[~, ~, lowerbound_LB, upperbound_LB, upperbound__LB, loss_LB, obf_matrix_LB] = bendersDecomposition(masteragent, agent, env_parameters, ITER); 
time_LB_ep4=toc;
loss_matrix_LB=env_parameters.cost_matrix.*obf_matrix_LB;
loss_LB_ep4=sum(loss_matrix_LB(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps4_after_benders_lb', panda_mem_t0, ...
    {'agent','masteragent','lowerbound_LB','upperbound_LB','upperbound__LB','obf_matrix_LB','loss_matrix_LB'}, panda_mem_csv, panda_mem_mat);


%time_2PPO
ep=min(env_parameters.EPSILON,epsilon_nmw);


phase1_budget=mean(ep(:));
safety_margin=mean(privacy_budget(:));


%% 2PPO epsilon=7
% Cluster the nodes
env_parameters.EPSILON=7;
env_parameters.NR_AGENT=25;
cluster_idx = kmeans(distance_matrix, env_parameters.NR_AGENT); 
% Create the agents
%env_parameters.NEIGHBOR_THRESHOLD=1;
%fprintf('------------------- Create the agents ----------------------- \n'); 
tic;
agent_2PPO = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
%fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 

% Create the master agent
%fprintf('------------------- Create the agents ----------------------- \n'); 
masteragent  = masterAgentCreation(distance_matrix, agent_2PPO, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
time2=toc;
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps7_after_agent_master_2ppo', panda_mem_t0, ...
    {'cluster_idx','agent_2PPO','masteragent'}, panda_mem_csv, panda_mem_mat);

% The algorithm starts here!!
tic;
ITER = 100; 
[~, ~, lowerbound, upperbound, upperbound_, loss1, obf_matrix] = bendersDecomposition(masteragent, agent_2PPO, env_parameters, ITER); 
time_2PPO_ep7=toc;
loss_matrix=env_parameters.cost_matrix.*obf_matrix;
loss_ep7=sum(loss_matrix(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps7_after_benders_2ppo', panda_mem_t0, ...
    {'agent_2PPO','masteragent','lowerbound','upperbound','upperbound_','obf_matrix','loss_matrix'}, panda_mem_csv, panda_mem_mat);



%% LB epsilon=7
env_parameters.EPSILON=7;
agent = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
masteragent  = masterAgentCreation(distance_matrix, agent, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps7_after_agent_master_lb', panda_mem_t0, ...
    {'agent','masteragent'}, panda_mem_csv, panda_mem_mat);
tic;
ITER = 100;
[~, ~, lowerbound_LB, upperbound_LB, upperbound__LB, loss_LB, obf_matrix_LB] = bendersDecomposition(masteragent, agent, env_parameters, ITER); 
time_LB_ep7=toc;
loss_matrix_LB=env_parameters.cost_matrix.*obf_matrix_LB;
loss_LB_ep7=sum(loss_matrix_LB(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps7_after_benders_lb', panda_mem_t0, ...
    {'agent','masteragent','lowerbound_LB','upperbound_LB','upperbound__LB','obf_matrix_LB','loss_matrix_LB'}, panda_mem_csv, panda_mem_mat);






%% 2PPO epsilon=10
% Cluster the nodes
env_parameters.EPSILON=10;
env_parameters.NR_AGENT=25;
cluster_idx = kmeans(distance_matrix, env_parameters.NR_AGENT); 
% Create the agents
%env_parameters.NEIGHBOR_THRESHOLD=1;
%fprintf('------------------- Create the agents ----------------------- \n'); 
tic;
agent_2PPO = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
%fprintf('%d agents have been created. \n', env_parameters.NR_AGENT); 

% Create the master agent
%fprintf('------------------- Create the agents ----------------------- \n'); 
masteragent  = masterAgentCreation(distance_matrix, agent_2PPO, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, epsilon_nmw, xi_hathat); 
time2=toc;
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps10_after_agent_master_2ppo', panda_mem_t0, ...
    {'cluster_idx','agent_2PPO','masteragent'}, panda_mem_csv, panda_mem_mat);

% The algorithm starts here!!
tic;
ITER = 100; 
[~, ~, lowerbound, upperbound, upperbound_, loss1, obf_matrix] = bendersDecomposition(masteragent, agent_2PPO, env_parameters, ITER); 
time_2PPO_ep10=toc;
loss_matrix=env_parameters.cost_matrix.*obf_matrix;
loss_ep10=sum(loss_matrix(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps10_after_benders_2ppo', panda_mem_t0, ...
    {'agent_2PPO','masteragent','lowerbound','upperbound','upperbound_','obf_matrix','loss_matrix'}, panda_mem_csv, panda_mem_mat);



%% LB epsilon=10
env_parameters.EPSILON=10;
agent = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, env_parameters.NR_AGENT, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
masteragent  = masterAgentCreation(distance_matrix, agent, adjacence_matrix, cluster_idx, env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC, env_parameters.NR_AGENT, env_parameters.EPSILON, 0*epsilon_nmw, 0*xi_hathat); 
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps10_after_agent_master_lb', panda_mem_t0, ...
    {'agent','masteragent'}, panda_mem_csv, panda_mem_mat);
tic;
ITER = 100;
[~, ~, lowerbound_LB, upperbound_LB, upperbound__LB, loss_LB, obf_matrix_LB] = bendersDecomposition(masteragent, agent, env_parameters, ITER); 
time_LB_ep10=toc;
loss_matrix_LB=env_parameters.cost_matrix.*obf_matrix_LB;
loss_LB_ep10=sum(loss_matrix_LB(:));
panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'eps10_after_benders_lb', panda_mem_t0, ...
    {'agent','masteragent','lowerbound_LB','upperbound_LB','upperbound__LB','obf_matrix_LB','loss_matrix_LB'}, panda_mem_csv, panda_mem_mat);

panda_mem_records = panda_mem_checkpoint(panda_mem_records, 'finish', panda_mem_t0, ...
    {'loss_ep4','loss_ep7','loss_ep10','loss_LB_ep4','loss_LB_ep7','loss_LB_ep10', ...
    'time_2PPO_ep4','time_2PPO_ep7','time_2PPO_ep10','time_LB_ep4','time_LB_ep7','time_LB_ep10'}, ...
    panda_mem_csv, panda_mem_mat);

function records = panda_mem_checkpoint(records, stage, t0, tracked_vars, csv_file, mat_file)
    [user_mem, sys_mem] = memory;
    tracked_bytes = panda_mem_tracked_bytes(tracked_vars);
    all_workspace_bytes = panda_mem_workspace_bytes();
    peak_so_far = user_mem.MemUsedMATLAB;
    if ~isempty(records)
        peak_so_far = max([records.matlab_mem_used_bytes; user_mem.MemUsedMATLAB]);
    end
    selected_n = panda_mem_eval_scalar('length(all_target)');
    current_n = panda_mem_eval_scalar('length(distance_matrix)');
    obf_k = panda_mem_eval_scalar('length(obf_loc)');
    epsilon = panda_mem_eval_scalar('env_parameters.EPSILON');
    nr_agent = panda_mem_eval_scalar('env_parameters.NR_AGENT');

    row = table(datetime("now"), string(stage), toc(t0), selected_n, current_n, obf_k, epsilon, nr_agent, ...
        user_mem.MemUsedMATLAB, peak_so_far, user_mem.MemAvailableAllArrays, user_mem.MaxPossibleArrayBytes, ...
        sys_mem.PhysicalMemory.Available, sys_mem.PhysicalMemory.Total, tracked_bytes, all_workspace_bytes, ...
        'VariableNames', {'timestamp','stage','elapsed_sec','selected_n','current_n','obf_k','epsilon','nr_agent', ...
        'matlab_mem_used_bytes','matlab_mem_peak_so_far_bytes','matlab_mem_available_all_arrays_bytes', ...
        'matlab_max_possible_array_bytes','physical_mem_available_bytes','physical_mem_total_bytes', ...
        'tracked_vars_bytes','workspace_bytes'});
    records = [records; row];
    writetable(records, csv_file);
    save(mat_file, 'records');
end

function bytes = panda_mem_tracked_bytes(var_names)
    bytes = 0;
    for idx = 1:numel(var_names)
        info = evalin('caller', sprintf('whos(''%s'')', var_names{idx}));
        if ~isempty(info)
            bytes = bytes + sum([info.bytes]);
        end
    end
end

function bytes = panda_mem_workspace_bytes()
    info = evalin('caller', 'whos');
    if isempty(info)
        bytes = 0;
    else
        bytes = sum([info.bytes]);
    end
end

function value = panda_mem_eval_scalar(expr)
    try
        value = evalin('caller', expr);
        if isempty(value) || ~isscalar(value) || ~isnumeric(value)
            value = NaN;
        end
    catch
        value = NaN;
    end
end
