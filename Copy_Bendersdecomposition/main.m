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

% rng("default")

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
% 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.95; 
% TARGET_LAT_MIN = 40.801;
% 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.8; 
% TARGET_LAT_MIN = 40.6501;
% 
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

% TARGET_LON_MAX = 0.3; 
% TARGET_LON_MIN = 0.101; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;
% ----------------------------
% ----------------------------

% env_parameters.longitude_min = TARGET_LON_MIN;
% env_parameters.longitude_max = TARGET_LON_MAX; 
% env_parameters.latitude_min = TARGET_LAT_MIN; 
% env_parameters.latitude_max = TARGET_LAT_MAX; 



Regions = [ ...
 %% ROME DATASET (FOR 2000 NODES)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.401,'lon_max',12.6,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.6,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.601,'lon_max',12.8,'lat_min',41.801,'lat_max',42.00) ];      

 %% ROME DATASET (4000 $ 6000 NODES)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.701,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.59,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.59,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.5901,'lon_max',12.8,'lat_min',41.801,'lat_max',42.00) ];

 %% ROME DATASET (8000 $ 10000 NODES)
 struct('lon_min',12.2,'lon_max',12.45,'lat_min',41.701,'lat_max',42.10)
 struct('lon_min',12.4501,'lon_max',12.55,'lat_min',41.701,'lat_max',42.10)
 struct('lon_min',12.5501,'lon_max',12.8,'lat_min',41.7501,'lat_max',42.00) ];


 %% NYC DATASET (2000, 4000 $ 6000 NODES)
 % struct('lon_min',-74.3,'lon_max',-74,'lat_min',40.5,'lat_max',40.65)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.801,'lat_max',40.95)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.6501,'lat_max',40.8)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.5,'lat_max',40.65) ]; 

 %% NYC DATASET (8000 $ 10000 NODES)
 % struct('lon_min',-74.3,'lon_max',-74,'lat_min',40.5,'lat_max',40.67)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.7801,'lat_max',40.95)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.6701,'lat_max',40.78)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.5,'lat_max',40.67) ]; 


 %% LONDON DATASET
 % struct('lon_min',-0.5,'lon_max',-0.3,'lat_min',51.4,'lat_max',51.6)
 % struct('lon_min',-0.301,'lon_max',-0.1,'lat_min',51.501,'lat_max',51.7)
 % struct('lon_min',-0.301,'lon_max',-0.1,'lat_min',51.3,'lat_max',51.5)
 % struct('lon_min',-0.101,'lon_max',0.1,'lat_min',51.501,'lat_max',51.7)
 % struct('lon_min',-0.101,'lon_max',0.1,'lat_min',51.3,'lat_max',51.5)
 % struct('lon_min',0.101,'lon_max',0.3,'lat_min',51.4,'lat_max',51.6) ];
R = numel(Regions);


LR_LOC_SIZE = 20;                                                           % The total number of locations
OBF_RANGE = 4.5;                                                            % The obfuscation range is considered as a circle, and OBF_RANGE is the radius
EXP_RANGE = 4.0;                                                            % The set of location not applying exponential mechanism is within a circle, of which the radius is EXP_RANGE. 
% OBF_RANGE = 20;                                                             % The obfuscation range is considered as a circle, and OBF_RANGE is the radius
% EXP_RANGE = 10;                                                             % The set of location not applying exponential mechanism is within a circle, of which the radius is EXP_RANGE. 
NEIGHBOR_THRESHOLD = 0.5;                                                   % The neighbor threshold eta
NR_DEST = 1;                                                                % The number of destinations (spatial tasks)
NR_USER = 10;                                                               % The number of users (agents)
LR_SAMPLE_SIZE = 100;                                                       % ??Michael
NR_LOC = 1;


% Initialize arrays for storing metrics across R regions
cost             = nan(1, R);   
cost_test        = nan(1, R);   
computation_time = nan(1, R);  
nr_violations    = nan(1, R);  
violation_mag    = nan(1, R);  

PL_DLR   = nan(R, NR_USER);   % Privacy Loss for Original Distance Matrix
Rel_DLR  = nan(R, NR_USER);   % Relative Frobenius error for Original Distance Matrix
Viol_DLR = nan(R, NR_USER);   % Violation Ratio for Original Distance Matrix
Rel_DLR2  = nan(R, NR_USER);  % Relative Frobenius error for Obfuscated Distance Matrix
Viol_DLR2 = nan(R, NR_USER);  % Violation Ratio for Obfuscated Distance Matrix
Rel_CRL = nan(R, NR_USER);    % Relative Frobenius error for Cost Matrix
% -------------------------------  
Rel_DLR_p   = nan(R, NR_USER);   
Viol_DLR_p  = nan(R, NR_USER);   
Rel_DLR2_p  = nan(R, NR_USER);  
Viol_DLR2_p = nan(R, NR_USER);     
Rel_CRL_p   = nan(R, NR_USER);    
% -------------------------------
Rel_DLR_r   = nan(R, NR_USER);   
Viol_DLR_r  = nan(R, NR_USER);  
Rel_DLR2_r  = nan(R, NR_USER);  
Viol_DLR2_r = nan(R, NR_USER);   
Rel_CRL_r   = nan(R, NR_USER);    
% -------------------------------
Rel_DLR_s   = nan(R, NR_USER);   
Viol_DLR_s  = nan(R, NR_USER);  
Rel_DLR2_s  = nan(R, NR_USER);  
Viol_DLR2_s = nan(R, NR_USER);     
Rel_CRL_s   = nan(R, NR_USER);    



baseSeed = 12345;                                   % any fixed number you like
stream   = RandStream('Threefry','Seed',baseSeed);  % or 'mrg32k3a' if you prefer
RandStream.setGlobalStream(stream);                 % make it the global RNG



EPSILON = 4; 

for r =  1:R     
    % --- set region bounds for this run ---
    env_parameters.longitude_min = Regions(r).lon_min;
    env_parameters.longitude_max = Regions(r).lon_max;
    env_parameters.latitude_min  = Regions(r).lat_min;
    env_parameters.latitude_max  = Regions(r).lat_max;

    env_parameters.nr_loc_selected = 500; 
   
    env_parameters.nr_loc_selected = NR_LOC*8000; 
    
    
    %% Initialization
    env_parameters = readCityMapInfo(env_parameters);                           % Create the road map information of the target region: Rome, Italy
    % env_parameters = readGridMapInfo(env_parameters);                         % Create the road map information of the target region: Rome, Italy
    env_parameters.GAMMA = 1000; 
    env_parameters.NEIGHBOR_THRESHOLD = 50;
 

    %% Create the server
    server = Server(NR_DEST, EXP_RANGE, CRT_GRID_CELL_SIZE);                    % Create the server

      
    %% Create the users        
    for m = 1:1:NR_USER

        stream.Substream = (r-1)*NR_USER + m;          % 1..(R*NR_USER)
        RandStream.setGlobalStream(stream);

        idx_selected = randperm(size(env_parameters.node_target, 2), env_parameters.nr_loc_selected); 
        env_parameters.longitude_selected = env_parameters.longitude(idx_selected); 
        env_parameters.latitude_selected = env_parameters.latitude(idx_selected);
        env_parameters.node_target_selected = env_parameters.node_target(idx_selected); 
        env_parameters.G_mDP = mDP_graph_creator(env_parameters);

        user(m, 1) = User(m, LR_LOC_SIZE, OBF_RANGE, NEIGHBOR_THRESHOLD, env_parameters);               % Create users
        user(m, 1) = user(m, 1).initialization(env_parameters);                                         % Initialize the properties of the user, including the local relevant locations, distance matrices, obfuscated location IDs, and the cost matrix
        

        lon_sel    = env_parameters.longitude_selected;
        lat_sel    = env_parameters.latitude_selected;
        node_tar   = env_parameters.node_target_selected;
        LR_ID      = user(m,1).LR_loc_ID;
        obf_ID     = user(m,1).obf_loc_ID;
        cost_matrix= user(m,1).cost_matrix_RL;

        % unique filename per region/user to avoid overwrite
        fname = sprintf('location_data_r%d_user%d.mat', r, m);
        save(fname, 'lon_sel','lat_sel','node_tar','LR_ID','obf_ID','cost_matrix','-v7.3');    
    end 
        


    % for m = 1:NR_USER
    %     % make region-specific subfolders
    %     outG = sprintf('london_fig_surfaces_6000/r%02d/g', r);
    %     outP = sprintf('london_fig_surfaces_6000/r%02d/p', r);
    %     outR = sprintf('london_fig_surfaces_6000/r%02d/rbf', r);
    %     outS = sprintf('london_fig_surfaces_6000/r%02d/svd', r);
    % 
    %     if ~exist(outG,'dir'), mkdir(outG); end
    %     if ~exist(outP,'dir'), mkdir(outP); end
    %     if ~exist(outR,'dir'), mkdir(outR); end
    %     if ~exist(outS,'dir'), mkdir(outS); end
    % 
    %     % pass a full, user-specific stem to avoid collisions
    %     stemG = sprintf('%s/fig_surfaces_g_r%02d_u%02d', outG, r, m);
    %     stemP = sprintf('%s/fig_surfaces_p_r%02d_u%02d', outP, r, m);
    %     stemR = sprintf('%s/fig_surfaces_r_r%02d_u%02d', outR, r, m);
    %     stemS = sprintf('%s/fig_surfaces_s_r%02d_u%02d', outS, r, m);
    % 
    %     plot_all_surfaces_for_user_g(user(m), stemG);
    %     plot_all_surfaces_for_user_p(user(m), stemP);
    %     plot_all_surfaces_for_user_r(user(m), stemR);
    %     plot_all_surfaces_for_user_s(user(m), stemS);
    % end




    for m = 1:NR_USER
        u = user(m); % (NR_USER=10 in your script)

        if ~isempty(u.distance_matrix_LR_recovered)
            % [~, ~, PL_max_1] = compute_log_posterior(u.fitted_best_params_1, u.n_fitted_best_params_1, u.distance_matrix, size(u.distance_matrix_LR,1), 1/u.epsilon, size(u.distance_matrix_LR,1)); 
            % PL_DLR(r) = PL_max_1; 
            Rel_DLR(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered);
            Viol_DLR(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered);
        end    
        if ~isempty(u.distance_matrix_LR2obf_recovered)
            Rel_DLR2(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered);
            Viol_DLR2(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered);
        end    
        if ~isempty(u.cost_matrix_RL_recovered)
            Rel_CRL(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered);
        end
        % -------------------------------------------------------------------------------------------------------------
        if ~isempty(u.distance_matrix_LR_recovered_p)    
            Rel_DLR_p(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_p);
            Viol_DLR_p(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_p);
        end   
        if ~isempty(u.distance_matrix_LR2obf_recovered_p)
            Rel_DLR2_p(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_p);
            Viol_DLR2_p(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_p);
        end   
        if ~isempty(u.cost_matrix_RL_recovered_p)
            Rel_CRL_p(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_p);
        end
        % ------------------------------------------------------------------------------------------------------------
        if ~isempty(u.distance_matrix_LR_recovered_r)
            Rel_DLR_r(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_r);
            Viol_DLR_r(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_r);
        end    
        if ~isempty(u.distance_matrix_LR2obf_recovered_r)
            Rel_DLR2_r(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_r);
            Viol_DLR2_r(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_r);
        end    
        if ~isempty(u.cost_matrix_RL_recovered_r)
            Rel_CRL_r(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_r);
        end
        % ------------------------------------------------------------------------------------------------------------
        if ~isempty(u.distance_matrix_LR_recovered_s)
            Rel_DLR_s(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_s);
            Viol_DLR_s(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_s);
        end    
        if ~isempty(u.distance_matrix_LR2obf_recovered_s)
            Rel_DLR2_s(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_s);
            Viol_DLR2_s(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_s);
        end    
        if ~isempty(u.cost_matrix_RL_recovered_s)
            Rel_CRL_s(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_s);
        end
    end
end


% Collapse user dimension to region means (omit NaNs)
Rel_DLR_mean_by_region   = mean(Rel_DLR,  2, 'omitnan');
Viol_DLR_mean_by_region  = mean(Viol_DLR, 2, 'omitnan');
Rel_DLR2_mean_by_region  = mean(Rel_DLR2,  2, 'omitnan');
Viol_DLR2_mean_by_region = mean(Viol_DLR2, 2, 'omitnan');
Rel_CRL_mean_by_region   = mean(Rel_CRL, 2, 'omitnan');

% Then your overall means/stds across regions:
Mean_Rel_DLR  = mean(Rel_DLR_mean_by_region,  'omitnan');   CI95_Rel_DLR  = 1.96 * std(Rel_DLR_mean_by_region,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region));
Mean_Vio_DLR  = mean(Viol_DLR_mean_by_region, 'omitnan');   CI95_Vio_DLR  = 1.96 * std(Viol_DLR_mean_by_region, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region));
Mean_Rel_DLR2 = mean(Rel_DLR2_mean_by_region, 'omitnan');   CI95_Rel_DLR2 = 1.96 * std(Rel_DLR2_mean_by_region, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region));
Mean_Vio_DLR2 = mean(Viol_DLR2_mean_by_region,'omitnan');   CI95_Vio_DLR2 = 1.96 * std(Viol_DLR2_mean_by_region,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region));
Mean_Rel_CRL  = mean(Rel_CRL_mean_by_region,  'omitnan');   CI95_Rel_CRL  = 1.96 * std(Rel_CRL_mean_by_region,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region));

Summary = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR;  Mean_Vio_DLR;  Mean_Rel_DLR2;  Mean_Vio_DLR2;  Mean_Rel_CRL], ...
    [CI95_Rel_DLR;    CI95_Vio_DLR;    CI95_Rel_DLR2;    CI95_Vio_DLR2;    CI95_Rel_CRL], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);

disp(Summary);
writetable(Summary, 'Summary_Gaussian.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_p   = mean(Rel_DLR_p,  2, 'omitnan');
Viol_DLR_mean_by_region_p  = mean(Viol_DLR_p, 2, 'omitnan');
Rel_DLR2_mean_by_region_p  = mean(Rel_DLR2_p,  2, 'omitnan');
Viol_DLR2_mean_by_region_p = mean(Viol_DLR2_p, 2, 'omitnan');
Rel_CRL_mean_by_region_p   = mean(Rel_CRL_p, 2, 'omitnan');

Mean_Rel_DLR_p  = mean(Rel_DLR_mean_by_region_p,  'omitnan');   CI95_Rel_DLR_p  = 1.96 * std(Rel_DLR_mean_by_region_p,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_p));
Mean_Vio_DLR_p  = mean(Viol_DLR_mean_by_region_p, 'omitnan');   CI95_Vio_DLR_p  = 1.96 * std(Viol_DLR_mean_by_region_p, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_p));
Mean_Rel_DLR2_p = mean(Rel_DLR2_mean_by_region_p, 'omitnan');   CI95_Rel_DLR2_p = 1.96 * std(Rel_DLR2_mean_by_region_p, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_p));
Mean_Vio_DLR2_p = mean(Viol_DLR2_mean_by_region_p,'omitnan');   CI95_Vio_DLR2_p = 1.96 * std(Viol_DLR2_mean_by_region_p,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_p));
Mean_Rel_CRL_p  = mean(Rel_CRL_mean_by_region_p,  'omitnan');   CI95_Rel_CRL_p  = 1.96 * std(Rel_CRL_mean_by_region_p,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_p));

Summary_p = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_p;  Mean_Vio_DLR_p;  Mean_Rel_DLR2_p;  Mean_Vio_DLR2_p;  Mean_Rel_CRL_p], ...
    [CI95_Rel_DLR_p;    CI95_Vio_DLR_p;    CI95_Rel_DLR2_p;    CI95_Vio_DLR2_p;    CI95_Rel_CRL_p], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_p);
writetable(Summary_p, 'Summary_Polynomial.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_r   = mean(Rel_DLR_r,  2, 'omitnan');
Viol_DLR_mean_by_region_r  = mean(Viol_DLR_r, 2, 'omitnan');
Rel_DLR2_mean_by_region_r  = mean(Rel_DLR2_r,  2, 'omitnan');
Viol_DLR2_mean_by_region_r = mean(Viol_DLR2_r, 2, 'omitnan');
Rel_CRL_mean_by_region_r   = mean(Rel_CRL_r, 2, 'omitnan');

Mean_Rel_DLR_r  = mean(Rel_DLR_mean_by_region_r,  'omitnan');   CI95_Rel_DLR_r  = 1.96 * std(Rel_DLR_mean_by_region_r,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_r));
Mean_Vio_DLR_r  = mean(Viol_DLR_mean_by_region_r, 'omitnan');   CI95_Vio_DLR_r  = 1.96 * std(Viol_DLR_mean_by_region_r, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_r));
Mean_Rel_DLR2_r = mean(Rel_DLR2_mean_by_region_r, 'omitnan');   CI95_Rel_DLR2_r = 1.96 * std(Rel_DLR2_mean_by_region_r, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_r));
Mean_Vio_DLR2_r = mean(Viol_DLR2_mean_by_region_r,'omitnan');   CI95_Vio_DLR2_r = 1.96 * std(Viol_DLR2_mean_by_region_r,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_r));
Mean_Rel_CRL_r  = mean(Rel_CRL_mean_by_region_r,  'omitnan');   CI95_Rel_CRL_r  = 1.96 * std(Rel_CRL_mean_by_region_r,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_r));

Summary_r = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_r;  Mean_Vio_DLR_r;  Mean_Rel_DLR2_r;  Mean_Vio_DLR2_r;  Mean_Rel_CRL_r], ...
    [CI95_Rel_DLR_r;    CI95_Vio_DLR_r;    CI95_Rel_DLR2_r;    CI95_Vio_DLR2_r;    CI95_Rel_CRL_r], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_r);
writetable(Summary_r, 'Summary_RBF.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_s   = mean(Rel_DLR_s,  2, 'omitnan');
Viol_DLR_mean_by_region_s  = mean(Viol_DLR_s, 2, 'omitnan');
Rel_DLR2_mean_by_region_s  = mean(Rel_DLR2_s,  2, 'omitnan');
Viol_DLR2_mean_by_region_s = mean(Viol_DLR2_s, 2, 'omitnan');
Rel_CRL_mean_by_region_s   = mean(Rel_CRL_s, 2, 'omitnan');

Mean_Rel_DLR_s  = mean(Rel_DLR_mean_by_region_s,  'omitnan');   CI95_Rel_DLR_s  = 1.96 * std(Rel_DLR_mean_by_region_s,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_s));
Mean_Vio_DLR_s  = mean(Viol_DLR_mean_by_region_s, 'omitnan');   CI95_Vio_DLR_s  = 1.96 * std(Viol_DLR_mean_by_region_s, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_s));
Mean_Rel_DLR2_s = mean(Rel_DLR2_mean_by_region_s, 'omitnan');   CI95_Rel_DLR2_s = 1.96 * std(Rel_DLR2_mean_by_region_s, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_s));
Mean_Vio_DLR2_s = mean(Viol_DLR2_mean_by_region_s,'omitnan');   CI95_Vio_DLR2_s = 1.96 * std(Viol_DLR2_mean_by_region_s,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_s));
Mean_Rel_CRL_s  = mean(Rel_CRL_mean_by_region_s,  'omitnan');   CI95_Rel_CRL_s  = 1.96 * std(Rel_CRL_mean_by_region_s,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_s));

Summary_s = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_s;  Mean_Vio_DLR_s;  Mean_Rel_DLR2_s;  Mean_Vio_DLR2_s;  Mean_Rel_CRL_s], ...
    [CI95_Rel_DLR_s;    CI95_Vio_DLR_s;    CI95_Rel_DLR2_s;    CI95_Vio_DLR2_s;    CI95_Rel_CRL_s], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_s);
writetable(Summary_s, 'Summary_SVD.csv');





% Mean_Rel_DLR_s_  = mean(Rel_DLR_s(:),  'omitnan');   CI95_Rel_DLR_s_  = std(Rel_DLR_s(:),  0, 'omitnan');
% Mean_Vio_DLR_s_  = mean(Viol_DLR_s(:), 'omitnan');   CI95_Vio_DLR_s_  = std(Viol_DLR_s(:), 0, 'omitnan');
% Mean_Rel_DLR2_s_ = mean(Rel_DLR2_s(:), 'omitnan');   CI95_Rel_DLR2_s_ = std(Rel_DLR2_s(:), 0, 'omitnan');
% Mean_Vio_DLR2_s_ = mean(Viol_DLR2_s(:),'omitnan');   CI95_Vio_DLR2_s_ = std(Viol_DLR2_s(:),0, 'omitnan');
% Mean_Rel_CRL_s_  = mean(Rel_CRL_s(:),  'omitnan');   CI95_Rel_CRL_s_  = std(Rel_CRL_s(:),  0, 'omitnan');
% 
% Summary_s_ = table( ...
%     ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
%     [Mean_Rel_DLR_s_;  Mean_Vio_DLR_s_;  Mean_Rel_DLR2_s_;  Mean_Vio_DLR2_s_;  Mean_Rel_CRL_s_], ...
%     [CI95_Rel_DLR_s_;    CI95_Vio_DLR_s_;    CI95_Rel_DLR2_s_;    CI95_Vio_DLR2_s_;    CI95_Rel_CRL_s_], ...
%     'VariableNames', {'Metric','Mean','StdDev'} ...
% );
% disp(Summary_s_);
% writetable(Summary_s_, 'Summary_SVD_.csv');