addpath('./functions/');
% parameters; 

tic;      
%% Load the map dataset
opts = detectImportOptions('./datasets/rome/rome_nodes.csv');
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable('./datasets/rome/rome_nodes.csv', opts);
df_edges = readtable('./datasets/rome/rome_edges.csv');


% opts = detectImportOptions('./datasets/nyc/nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./datasets/nyc/nodes.csv', opts);
% df_edges = readtable('./datasets/nyc/edges.csv');


% opts = detectImportOptions('./datasets/london/nodes.csv');
% opts = setvartype(opts, 'osmid', 'int64');
% df_nodes = readtable('./datasets/london/nodes.csv', opts);
% df_edges = readtable('./datasets/london/edges.csv');


% Extract relevant columns
col_longitude = table2array(df_nodes(:, 'x'));  % Actual x (longitude) coordinate
col_latitude = table2array(df_nodes(:, 'y'));   % Actual y (latitude) coordinate
col_osmid = table2array(df_nodes(:, 'osmid')); 
env_parameters.NR_LOC = size(col_longitude, 1);


% Debug: Print min/max longitude and latitude
disp('Longitude and Latitude inside Target Region:');
disp([col_longitude, col_latitude]);
fprintf("Longitude Range: [%.6f, %.6f]\n", min(col_longitude), max(col_longitude));
fprintf("Latitude Range:  [%.6f, %.6f]\n", min(col_latitude), max(col_latitude));


% Define target region bounds 
%% ROME DATASET
% ----------------------------
% ----------------------------
% TARGET REGIONS (5)
% ----------------------------
% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1;
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;
% ----------------------------
% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;
% ----------------------------
% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.601; 
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.801;
% ----------------------------

% WHOLE TARGET REGION
% ----------------------------
% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 41.65;

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
%% NYC DATASET
% ----------------------------
% ----------------------------
% TARGET REGIONS (4)
% ----------------------------
% TARGET_LON_MAX = -74; 
% TARGET_LON_MIN = -74.3; 
% TARGET_LAT_MAX = 40.65; 
% TARGET_LAT_MIN = 40.5;
% ----------------------------
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

% WHOLE TARGET REGION
% ----------------------------
% TARGET_LON_MAX = -73.700473; 
% TARGET_LON_MIN = -74.254901; 
% TARGET_LAT_MAX = 40.912507; 
% TARGET_LAT_MIN = 40.498385;


%% LONDON DATASET
% ----------------------------
% ----------------------------
% TARGET REGIONS (12)
% ----------------------------
% TARGET_LON_MAX = -0.3; 
% TARGET_LON_MIN = -0.5; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;
% ----------------------------
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
% ----------------------------
% TARGET_LON_MAX = 0.3; 
% TARGET_LON_MIN = 0.101; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;
% ----------------------------

% WHOLE TARGET REGION
% ----------------------------
% TARGET_LON_MAX = 0.4; 
% TARGET_LON_MIN = -0.6; 
% TARGET_LAT_MAX = 51.7;
% TARGET_LAT_MIN = 51.25;
% ----------------------------
% ----------------------------

pause(1);
time_data = toc;

 
% tic;   
% % Call function to find nodes within the target region
% [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN);
% 
% pause(2);
% time_select_target_nodes = toc;
 
 
tic;   
% % Generate obfuscated locations as the first 40% of the selected points
% num_obfuscated = max(1, floor(0.1 * length(original_longitude))); % Ensure at least one point is selected
% obfuscated_longitude = original_longitude(1:num_obfuscated);
% obfuscated_latitude = original_latitude(1:num_obfuscated);
% 
% target_lat = col_latitude_selected(1);        % Target latitude
% target_long = col_longitude_selected(1);      % Target longitude
% % -----------------------------------------------------------------------------


%% --- Reuse saved selection/IDs instead of recomputing ---
S = load('Bendersdecomposition/rome_location_data_2000_nodes/location_data_sample_1/location_data_r1_user1.mat', ...
         'lon_sel','lat_sel','node_tar','obf_ID');   % add 'LR_ID' if you need it later

% Treat the saved selected set as your "original" set
col_osmid_selected  = S.node_tar;        % if you still print/use osmid list
col_longitude_selected  = S.lon_sel(:);      % for plotting
col_latitude_selected   = S.lat_sel(:);

num_points = min(200, numel(S.lon_sel));   
original_longitude  = S.lon_sel(1:num_points);
original_latitude   = S.lat_sel(1:num_points);

R = load('Bendersdecomposition/rome_location_data_2000_nodes/location_data_sample_1/location_data_r1_user1.mat','LR_ID');
nearest_longitude = S.lon_sel(R.LR_ID);
nearest_latitude  = S.lat_sel(R.LR_ID);

% Obfuscated = the indices in obf_ID into the selected set
obf_idx = S.obf_ID(:);
obfuscated_longitude = S.lon_sel(obf_idx);
obfuscated_latitude  = S.lat_sel(obf_idx);

% If you still want a “target” location, just pick one from the selected set
target_long = original_longitude(1);
target_lat  = original_latitude(1);

% Update counts that downstream code relies on
env_parameters.NR_LOC = numel(original_longitude);
num_locations = numel(original_longitude);
num_obf       = numel(obfuscated_longitude);



% Debug: Print the number of original and obfuscated nodes
fprintf("Number of selected nodes: %d\n", length(col_osmid_selected));
disp('Col_Osmid_Selected:');
disp(col_osmid_selected);



fprintf("Number of original nodes: %d\n", length(original_longitude))
disp('Original Locations:');
disp([original_longitude, original_latitude]);



fprintf("Number of obfuscated nodes: %d\n", length(obfuscated_longitude));
disp('Obfuscated Locations:');
disp([obfuscated_longitude, obfuscated_latitude]);



disp('Target Location:');
disp([target_long, target_lat]);





% The following plot is just for testing 
figure;
plot(col_longitude, col_latitude, 'o'); 
hold on; 
plot(col_longitude_selected, col_latitude_selected, 'bs', 'MarkerFaceColor', 'b'); % Highlight selected nodes
plot(original_longitude, original_latitude, 'bs', 'MarkerFaceColor', 'm'); % Highlight selected nodes
plot(nearest_longitude, nearest_latitude, 'ro', 'MarkerFaceColor', 'r'); % Highlight original nodes
plot(obfuscated_longitude, obfuscated_latitude, 'bs', 'MarkerFaceColor', 'g'); % Highlight obfuscated nodes
plot(target_long, target_lat, 'bs', 'MarkerFaceColor', 'y'); % Highlight target node
xlabel('Longitude');
ylabel('Latitude');
title('Selected, Original & Obfuscated Nodes & Target Nodes in Target Region');
grid on;
hold off;



% Compute pairwise Haversine distances and build raw and noisy distance matrices for each location
% num_locations = length(original_longitude);  % Number of locations
% num_obf = length(obfuscated_longitude);  % Number of obfuscated locations

original_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

obfuscated_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

cost_distance_matrix_original = cell(num_locations, 1);  % Cell array to store the 10x10 matrices
% cost_distance_matrix_obfuscated = cell(num_obf, 1); 
cost_coefficient_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

F1_norm = cell(num_locations, 1);
F2_norm = cell(num_locations, 1);
F3_norm = cell(num_locations, 1);

noisy_distance_matrices = cell(num_locations, 1); % Cell array to store the noisy 10x10 matrices
perturbation_probabilities = cell(num_locations, 1); % Cell array to store perturbation probabilities
posterior_probabilities = cell(num_locations, 1); % Cell array to store posterior probabilities

pause(3);
time_info = toc;



tic;   
distance_matrix = compute_distance_matrix(original_latitude, original_longitude); 
% raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude);

pause(4);
time_compute_distance_matrix = toc;



%% Use captial letters for the constants 
EPSILON = 4;  % EPSILON value for noise
B_VALUE = 1/ EPSILON;  % B_VALUE for perturbation probability calculation
CARDINALITY_N = numel(nearest_longitude);
lambda2 = 1.0;
lambda3 = 1.0;






tic;   
%% Loop through all locations (A, B, C, ..., T) and create their original raw distance
for i = 1:num_locations
    % [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);

    % Compute the raw distance matrix before alignment
    raw_distance_matrix_unaligned = compute_distance_matrix([original_latitude(i); nearest_latitude], ...
                                                                [original_longitude(i); nearest_longitude]);


    % Store the aligned distance matrix in the cell array
    original_distance_matrices{i} = raw_distance_matrix_unaligned;


    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf('Original Distance Matrix:\n');
    disp(original_distance_matrices{i});
end

pause(5);
time_original_distance = toc;





tic;   
%% Loop through all locations (A, B, C, ..., T) and create their obfuscated raw distance
for i = 1:num_locations
    % Select the 9 nearest neighbors for the current location
    % [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);


    % Compute the raw distance matrix before alignment
    raw_distance_matrix_unaligned = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], ...
                                                                [original_latitude(i); nearest_latitude], ...
                                                                obfuscated_longitude, ...
                                                                obfuscated_latitude);

    
    % Store the aligned distance matrix in the cell array
    obfuscated_distance_matrices{i} = raw_distance_matrix_unaligned;


    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf('Obfuscated Distance Matrix:\n');
    disp(obfuscated_distance_matrices{i});
end

pause(6);
time_obfuscated_distance = toc;





tic;   
%% Loop through all locations (A, B, C, ..., T) and create their raw distance/cost matrices
for i = 1:num_locations
    % Select the 9 nearest neighbors for the current location
    % [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);
    
    % Compute pairwise shortest path distances between original and target locations before alignment
    raw_distance_matrix_original_unaligned = compute_raw_distance_matrix_cost([original_longitude(i); nearest_longitude], ...
                                                            [original_latitude(i); nearest_latitude], ...
                                                            target_long, ...
                                                            target_lat, ...
                                                            df_edges, ...
                                                            df_nodes);


    % Store the aligned distance matrix in the cell array
    cost_distance_matrix_original{i} = raw_distance_matrix_original_unaligned;


    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf(' Original Cost Distance Matrix (Original vs Target):\n');
    disp(cost_distance_matrix_original{i});
end

pause(7);
time_cost_original_distance = toc;



tic;   
% Compute pairwise shortest path distances between obfuscated and target locations
cost_distance_matrix_obfuscated = compute_raw_distance_matrix_cost(obfuscated_longitude, obfuscated_latitude, target_long, target_lat, df_edges, df_nodes);

fprintf('Obfuscated Cost Distance Matrix (Obfuscated vs Target):\n');
disp(cost_distance_matrix_obfuscated);

pause(8);
time_cost_obfuscated_distance = toc;




tic;   
for idx = 1:num_locations
    % It is assumed to be a vector of length CARDINALITY_N.
    original_distance_vector = cost_distance_matrix_original{idx};

    % cost_coefficient = zeros(CARDINALITY_N, num_obf);
    n = numel(original_distance_vector);                              
    cost_coefficient = zeros(n, num_obf); 

    % Compute the cost coefficient matrix element-wise. Loop over each element in the original vector (neighbors) and each obfuscated distance.
    for i = 1:n
        for j = 1:num_obf
            cost_coefficient(i, j) = abs(original_distance_vector(i) - cost_distance_matrix_obfuscated(j));
        end
    end

    cost_coefficient_matrices{idx} = cost_coefficient;
    
    % Display the results for the current location.
    fprintf('Results for Location %d:\n', idx);
    fprintf('Cost Coefficient Matrix (Modulus of Differences):\n');
    disp(cost_coefficient_matrices{idx});
end

 pause(9);
time_cost_distance = toc;





tic;   
stats_store  = cell(num_locations,1);
iter_count   = zeros(num_locations,1);
converged    = false(num_locations,1);
t_total      = zeros(num_locations,1);
t_fit_init   = zeros(num_locations,1);
t_swap_sum   = zeros(num_locations,1);
t_fit_sum    = zeros(num_locations,1);        % fminsearch inside loop only
t_fit_all    = zeros(num_locations,1);        % fit_init + fit_sum
swap_evals_total   = zeros(num_locations,1);
swap_accepts_total = zeros(num_locations,1);
stop_reason  = cell(num_locations,1);



% preallocate
best_pi      = cell(num_locations,1);
best_params  = cell(num_locations,1);

GF1 = cell(num_locations, 1);
GF2 = cell(num_locations, 1);
GF3 = cell(num_locations, 1);
utility_loss = zeros(num_locations, 1);
differences = zeros(num_locations, 1);

for i = 1:num_locations
    A1 = original_distance_matrices{i};
    A2 = obfuscated_distance_matrices{i};
    A3 = cost_coefficient_matrices{i};

    % Call your reordering and visualization function
    % [best_pi{i}, best_params{i}, GF1{i}, GF2{i}, GF3{i}] = reorder_fit_gaussians(A1, A2, A3, 1.0, 1.0);

    [best_pi{i}, best_params{i}, GF1{i}, GF2{i}, GF3{i}, stats_i] = reorder_fit_gaussians_(A1, A2, A3, 1.0, 1.0);
    
    stats_store{i} = stats_i;
    iter_count(i)  = stats_i.outer_iters;
    converged(i)   = stats_i.converged;
    stop_reason{i} = stats_i.stop_reason;
    t_total(i)     = stats_i.time_total;
    t_fit_init(i)  = stats_i.time_fit_init;
    t_swap_sum(i)  = sum(stats_i.time_swap);
    t_fit_sum(i)   = sum(stats_i.time_fit);
    t_fit_all(i)   = stats_i.time_fit_init + sum(stats_i.time_fit);
    swap_evals_total(i)   = sum(stats_i.swap_evals);
    swap_accepts_total(i) = sum(stats_i.swap_accepts);

    
    F1_norm{i} = best_params{i}(1, 1:6); 
    F2_norm{i} = best_params{i}(1, 7:12); 
    F3_norm{i} = best_params{i}(1, 13:18); 


    % Utility loss
    B3 = GF3{i};

    min_rows = min(size(A3,1), size(B3,1));
    min_cols = min(size(A3,2), size(B3,2));    
    A3_trimmed = A3(1:min_rows, 1:min_cols);
    B3_trimmed = B3(1:min_rows, 1:min_cols);

    utility_loss(i) = norm(A3_trimmed - B3_trimmed, 'fro');
end


swap_frac = 100 * (t_swap_sum ./ t_total);
fit_frac  = 100 * (t_fit_all ./ t_total);
Results = table( (1:num_locations)', iter_count, converged, stop_reason, ...
                 t_total, t_swap_sum, t_fit_init, t_fit_sum, t_fit_all, ...
                 swap_frac, fit_frac, swap_evals_total, swap_accepts_total, ...
                 'VariableNames', { ...
                    'Location','OuterIters','Converged','StopReason', ...
                    'TimeTotal_s','TimeSwap_s','TimeFitInit_s','TimeFitLoop_s','TimeFitAll_s', ...
                    'SwapFrac_pct','FitFrac_pct','SwapEvals','SwapAccepts'});
disp(Results);
writetable(Results, 'time_efficiency_gaussian.csv');




fprintf('\n=== Alternating Method Time-Efficiency Summary ===\n');
fprintf('Locations: %d\n', num_locations);
fprintf('Convergence rate: %.1f%%\n', 100*mean(converged));

fprintf('\n-- Total runtime (per location) --\n');
fprintf('Mean:   %.4f s\n', mean(t_total));
fprintf('Median: %.4f s\n', median(t_total));
fprintf('90%%:   %.4f s\n', prctile(t_total,90));
fprintf('Max:    %.4f s\n', max(t_total));

fprintf('\n-- Swap phase runtime (sum over outer iters) --\n');
fprintf('Mean:   %.4f s | Fraction mean: %.1f%%\n', mean(t_swap_sum), mean(swap_frac));
fprintf('Median: %.4f s\n', median(t_swap_sum));

fprintf('\n-- Fit runtime (init + loop) --\n');
fprintf('Mean:   %.4f s | Fraction mean: %.1f%%\n', mean(t_fit_all), mean(fit_frac));
fprintf('Median: %.4f s\n', median(t_fit_all));

fprintf('\n-- Outer iterations to converge (or hit max_iter) --\n');
fprintf('Mean:   %.2f\n', mean(iter_count));
fprintf('Median: %.2f\n', median(iter_count));
fprintf('90%%:   %.2f\n', prctile(iter_count,90));

t_per_iter = t_total ./ max(iter_count, 1);
fprintf('Mean time per executed outer iter: %.4f s\n', mean(t_per_iter));



% figure;
% boxplot([t_total, t_swap_sum, t_fit_all], ...
%         'Labels', {'Total','Swap sum','Fit (init+loop)'});
% ylabel('Seconds');
% title('Runtime Distribution Across Locations');
% grid on;


figure;
bar([swap_frac, fit_frac], 'stacked');
xlabel('Location');
ylabel('Percent of total runtime');
title('Runtime Fraction: Swap vs Parameter Fitting');
legend('Swap','Fit','Location','best');
grid on;


figure;
histogram(iter_count);
xlabel('Outer iterations executed');
ylabel('Count');
title('Iterations to Stop (Converged or Max Iter)');
grid on;


% figure;
% scatter(iter_count, t_total, 'filled');
% xlabel('Outer iterations');
% ylabel('Total runtime (s)');
% title('Total Runtime vs Iterations');
% grid on;




% for loc = 1:3
%     lh = stats_store{loc}.loss_hist;   % if you store stats per location
%     iters = 0:(numel(lh)-1);
% 
%     figure;
%     plot(iters, lh, '-o', 'LineWidth', 2);
%     xlabel('Outer iteration');
%     ylabel('loss\_current (after refit)');
%     title(sprintf('Loss vs Iteration (Location %d)', loc));
%     grid on;
% end


maxLen = 0;
for i = 1:num_locations
    maxLen = max(maxLen, numel(stats_store{i}.loss_hist));
end
LH = nan(num_locations, maxLen);
for i = 1:num_locations
    lh = stats_store{i}.loss_hist;
    LH(i,1:numel(lh)) = lh;
end
avg_lh = nanmean(LH, 1);
iters = 0:(numel(avg_lh)-1);
figure;
plot(iters, avg_lh, '-o', 'LineWidth', 2);
xlabel('Outer iteration');
ylabel('Average loss\_current (after refit)');
title('Average Loss vs Iteration (All Locations)');
grid on;




% for loc = 1:3
%     lh = stats_store{loc}.loss_hist;          % length = outer_iters+1
%     ls = stats_store{loc}.loss_swap_best;     % length = outer_iters
%     iters = 0:(numel(lh)-1);
% 
%     figure;
%     plot(iters, lh, '-o', 'LineWidth', 2); hold on;
%     plot(1:numel(ls), ls, '--s', 'LineWidth', 2);
%     xlabel('Outer iteration');
%     ylabel('Loss');
%     legend('After refit (loss\_hist)', 'Best during swaps (loss\_swap\_best)', 'Location','best');
%     title(sprintf('Loss vs Iteration (Location %d)', loc));
%     grid on; hold off;
% end


%====================
% for loc = 1:3
%     lh = stats_store{loc}.loss_hist;          % length = outer_iters+1 (includes iter 0)
%     ls = stats_store{loc}.loss_swap_best;     % length = outer_iters (iter 1..K)
%     lr = stats_store{loc}.loss_refit_only;    % length = outer_iters (iter 1..K)
% 
%     it_lh = 0:(numel(lh)-1);
%     it_k  = 1:numel(ls);
% 
%     figure;
%     plot(it_k, ls, '--s', 'LineWidth', 2); hold on;          % swap-only
%     plot(it_k, lr, ':d', 'LineWidth', 2);                    % refit-only
%     plot(it_lh, lh, '-o', 'LineWidth', 2);                   % swap+refit (actual)
% 
%     xlabel('Outer iteration');
%     ylabel('Loss');
%     legend('Best during swaps (swap-only)', ...
%            'Refit-only (no swaps)', ...
%            'After swap + refit (actual)', ...
%            'Location','best');
%     title(sprintf('Loss vs Iteration (Location %d)', loc));
%     grid on; hold off;
% end




% ===================== Average curves across ALL locations =====================

num_locations = numel(stats_store);

% ---- 1) swap-only best (iter 1..K) ----
maxLenS = 0;
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_swap_best')
        maxLenS = max(maxLenS, numel(stats_store{i}.loss_swap_best));
    end
end
LS = nan(num_locations, maxLenS);
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_swap_best')
        ls = stats_store{i}.loss_swap_best;
        LS(i,1:numel(ls)) = ls;
    end
end
avg_ls = nanmean(LS, 1);
iters_s = 1:numel(avg_ls);

% ---- 2) refit-only (iter 1..K) ----
maxLenR = 0;
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_refit_only')
        maxLenR = max(maxLenR, numel(stats_store{i}.loss_refit_only));
    end
end
LR = nan(num_locations, maxLenR);
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_refit_only')
        lr = stats_store{i}.loss_refit_only;
        LR(i,1:numel(lr)) = lr;
    end
end
avg_lr = nanmean(LR, 1);
iters_r = 1:numel(avg_lr);

% ---- 3) swap + refit actual (loss_hist: iter 0..K) ----
maxLenH = 0;
for i = 1:num_locations
    maxLenH = max(maxLenH, numel(stats_store{i}.loss_hist));
end
LH = nan(num_locations, maxLenH);
for i = 1:num_locations
    lh = stats_store{i}.loss_hist;
    LH(i,1:numel(lh)) = lh;
end
avg_lh = nanmean(LH, 1);
iters_h = 0:(numel(avg_lh)-1);

% ---- Plot all three on one figure ----
figure;
plot(iters_s, avg_ls, '--s', 'LineWidth', 2); hold on;
plot(iters_r, avg_lr, ':d',  'LineWidth', 2);
plot(iters_h, avg_lh, '-o',  'LineWidth', 2);

xlabel('Outer iteration');
ylabel('Average Loss');
legend('Best during swaps (swap-only)', ...
       'Refit-only (no swaps)', ...
       'After swap + refit (actual)', ...
       'Location','best');
title('Average Loss vs Iteration (All Locations)');
grid on; hold off;


% ---- Red rectangle (ZOOM REGION) ----
x_zoom = [14 20];
y_zoom = [4690 4825];

rectangle('Position', [x_zoom(1), y_zoom(1), ...
                       diff(x_zoom), diff(y_zoom)], ...
          'EdgeColor','r', 'LineWidth',1);
hold off;

% ===== Create zoomed inset =====
ax_inset = [0.46 0.3 0.41 0.35];        % [left bottom width height] [0.58 0.35 0.3 0.35];  
axes('Position', ax_inset);  

% Plot the same curves again
hold on; grid on;

plot(iters_s, avg_ls, '--s', 'LineWidth', 1.5);
plot(iters_r, avg_lr, ':d',  'LineWidth', 1.5);
plot(iters_h, avg_lh, '-o',  'LineWidth', 1.5);

% ---- Set zoom region (adjust these!) ----
xlim(x_zoom);          % iteration range (your red box region)          
ylim(y_zoom);          % loss range (tight zoom)
set(gca, 'FontSize', 7);

hold off;

% ---- Red border around inset ----
annotation('rectangle', ax_inset, ...
    'Color','r', 'LineWidth',0.7);

% -----------------------------------------------------------








% ===== 2x2 Figure: Time efficiency + convergence + loss curves =====

figure('Position', [200 200 1100 850]);

% Create 2x2 axes
ax1 = subplot(2,2,1);   % stacked bar: swap vs fit fraction
ax2 = subplot(2,2,2);   % histogram: iterations
ax3 = subplot(2,2,3);   % average loss_hist only
ax4 = subplot(2,2,4);   % 3-curve average loss (swap-only, refit-only, swap+refit)

% Common formatting helper (optional inline)
set([ax1 ax2 ax3 ax4], 'FontSize', 10, 'LineWidth', 1.0);

% ---- (1) Runtime fraction: swap vs fit (stacked bar) ----
axes(ax1);
bar([swap_frac, fit_frac], 'stacked');
xlabel('Location');
ylabel('% of total runtime');
title('Runtime Fraction: Swap vs Fit');
legend('Swap','Fit','Location','best');
grid on;

% ---- (2) Histogram: iterations executed ----
axes(ax2); 
histogram(iter_count);
ymax = max(ax2.Children.BinCounts);
ylim(ax2, [0, 1.1*ymax]);
xlabel('Outer iterations executed');
ylabel('Count');
title('Iterations to Stop');
grid on;

% ---- (3) Average loss after swap+refit only (loss_hist) ----
axes(ax3);
% Build avg_lh from loss_hist
num_locations = numel(stats_store);

maxLenH = 0;
for i = 1:num_locations
    maxLenH = max(maxLenH, numel(stats_store{i}.loss_hist));
end

LH = nan(num_locations, maxLenH);
for i = 1:num_locations
    lh = stats_store{i}.loss_hist;
    LH(i,1:numel(lh)) = lh;
end

avg_lh_only = nanmean(LH, 1);
iters_only  = 0:(numel(avg_lh_only)-1);

plot(iters_only, avg_lh_only, '-o', 'LineWidth', 2);
xlabel('Outer iteration');
ylabel('Avg loss (after refit)');
title('Avg Loss vs Iteration (swap+refit)');
grid on;

% ---- (4) Average three loss curves: swap-only, refit-only, swap+refit ----
axes(ax4); 

% ---- 1) swap-only best (iter 1..K) ----
maxLenS = 0;
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_swap_best')
        maxLenS = max(maxLenS, numel(stats_store{i}.loss_swap_best));
    end
end
LS = nan(num_locations, maxLenS);
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_swap_best')
        ls = stats_store{i}.loss_swap_best;
        LS(i,1:numel(ls)) = ls;
    end
end
avg_ls = nanmean(LS, 1);
iters_s = 1:numel(avg_ls);

% ---- 2) refit-only (iter 1..K) ----
maxLenR = 0;
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_refit_only')
        maxLenR = max(maxLenR, numel(stats_store{i}.loss_refit_only));
    end
end
LR = nan(num_locations, maxLenR);
for i = 1:num_locations
    if isfield(stats_store{i}, 'loss_refit_only')
        lr = stats_store{i}.loss_refit_only;
        LR(i,1:numel(lr)) = lr;
    end
end
avg_lr = nanmean(LR, 1);
iters_r = 1:numel(avg_lr);

% ---- 3) swap + refit actual (loss_hist: iter 0..K) ----
% (reuse avg_lh_only computed above)
avg_lh = avg_lh_only;
iters_h = iters_only;

hold on; grid on;
if ~isempty(avg_ls)
    plot(iters_s, avg_ls, '--s', 'LineWidth', 2);
end
if ~isempty(avg_lr) && any(~isnan(avg_lr))
    plot(iters_r, avg_lr, ':d', 'LineWidth', 2);
end
plot(iters_h, avg_lh, '-o', 'LineWidth', 2);

xlabel('Outer iteration');
ylabel('Average loss');
title('Avg Loss Decomposition');
legendEntries = {};
if ~isempty(avg_ls), legendEntries{end+1} = 'Swap-only (best during swaps)'; end
if ~isempty(avg_lr) && any(~isnan(avg_lr)), legendEntries{end+1} = 'Refit-only (no swaps)'; end
legendEntries{end+1} = 'Swap + refit (actual)';
legend(legendEntries, 'Location','best');


% ---- Red rectangle (ZOOM REGION) ----
x_zoom = [14 20];
y_zoom = [4690 4825];

rectangle('Position', [x_zoom(1), y_zoom(1), ...
                       diff(x_zoom), diff(y_zoom)], ...
          'EdgeColor','r', 'LineWidth',1);
hold off;

% ===== Create zoomed inset =====
ax_inset = [0.69 0.2 0.2 0.15];        % [left bottom width height] [0.58 0.35 0.3 0.35];  
axes('Position', ax_inset);  

% Plot the same curves again
hold on; grid on;

plot(iters_s, avg_ls, '--s', 'LineWidth', 1.5);
plot(iters_r, avg_lr, ':d',  'LineWidth', 1.5);
plot(iters_h, avg_lh, '-o',  'LineWidth', 1.5);

% ---- Set zoom region (adjust these!) ----
xlim(x_zoom);          % iteration range (your red box region)          
ylim(y_zoom);          % loss range (tight zoom)
set(gca, 'FontSize', 7);

hold off;

% ---- Red border around inset ----
annotation('rectangle', ax_inset, ...
    'Color','r', 'LineWidth',0.7);
% -----------------------------------------------------------



















M1 = vertcat(F1_norm{:});   % size is [num_locations × 4]
M2 = vertcat(F2_norm{:});
M3 = vertcat(F3_norm{:});

mn1 = min(M1,[],1);    mx1 = max(M1,[],1);
mn2 = min(M2,[],1);    mx2 = max(M2,[],1);
mn3 = min(M3,[],1);    mx3 = max(M3,[],1);

M1_norm = (M1 - mn1) ./ (mx1 - mn1);
M2_norm = (M2 - mn2) ./ (mx2 - mn2);
M3_norm = (M3 - mn3) ./ (mx3 - mn3);

F1 = mat2cell(M1_norm, ones(size(M1_norm,1),1), size(M1_norm,2));
F2 = mat2cell(M2_norm, ones(size(M2_norm,1),1), size(M2_norm,2));
F3 = mat2cell(M3_norm, ones(size(M3_norm,1),1), size(M3_norm,2));





% for i = 1:num_locations
%     [G1, G2, G3] = rebuild_from_best_params( ...
%                      original_distance_matrices, ...
%                      obfuscated_distance_matrices, ...
%                      cost_coefficient_matrices, ...
%                      best_params);
% end


% % % ======================================================
% for i = 1:num_locations
%     n = size(original_distance_matrices{i},1);
%     m = size(obfuscated_distance_matrices{i},2);
% 
%     [G1,G2,G3] = generate_gaussians(n, m, best_params{i});   % <-- new names!
%     disp(size(G2));   
% 
%     % Undo the permutation so they line up with raw matrices
%     inv_pi           = zeros(1,n);
%     inv_pi(best_pi{i}) = 1:n;
% 
%     G1_orig = G1(inv_pi,inv_pi);
%     G2_orig = G2(inv_pi,:);
%     G3_orig = G3(inv_pi,:);
% 
%     A1 = original_distance_matrices{i};
%     A2 = obfuscated_distance_matrices{i};
%     A3 = cost_coefficient_matrices{i};
% 
%     fprintf('params = %s\n', mat2str(best_params{i},3));
%     fprintf('std(G1_orig(:)) = %.3g\n', std(G1_orig(:)));
%     fprintf('min/max  G1_orig = %.3g / %.3g\n', min(G1_orig(:)), max(G1_orig(:)));
% 
%     figure, imagesc(G1_orig), axis equal tight, colorbar
%     title('Raw Gaussian before denorm')
% 
% 
% 
%     % Denormalise if you want to compare on the original scale
%     minA1 = min(A1(:));  maxA1 = max(A1(:));
%     minA2 = min(A2(:));  maxA2 = max(A2(:));
%     minA3 = min(A3(:));  maxA3 = max(A3(:));
% 
%     G1_denorm = G1_orig * (maxA1 - minA1) + minA1;
%     G2_denorm = G2_orig * (maxA2 - minA2) + minA2;
%     G3_denorm = G3_orig * (maxA3 - minA3) + minA3;
% 
%     % (optional) store for later or plot immediately
%     if i==1        % example sanity check only for first location
%         figure;
%         subplot(1,2,1); imagesc(original_distance_matrices{1});
%         title('A1 original'); colorbar;
%         subplot(1,2,2); imagesc(G1_denorm);
%         title('Reconstructed from best\_params'); colorbar;
%     end
% end 

pause(10);
time_gaussian_fit = toc;






tic;   
num_samples = 100;                                % Number of samples for each posterior probability

total_entries_all_1 = 0;
total_violations_all_1 = 0;
num_violations_per_location_1 = zeros(num_locations, 1);  % Per-location count
total_entries_per_location_1 = zeros(num_locations, 1);   % For completeness
percentage_violations_1 = zeros(num_locations, 1);  % Final percentage per location
violation_fractions_1 = cell(num_locations, 1);     % Store per-sample fractions

total_entries_all_2 = 0;
total_violations_all_2 = 0;
num_violations_per_location_2 = zeros(num_locations, 1);  % Per-location count
total_entries_per_location_2 = zeros(num_locations, 1);   % For completeness
percentage_violations_2 = zeros(num_locations, 1);  % Final percentage per location
violation_fractions_2 = cell(num_locations, 1);     % Store per-sample fractions


posterior_prob_sample_1 = cell(num_samples,1);
posterior_prob_sample_2 = cell(num_samples,1);
posterior_prob_sample_3 = cell(num_samples,1);

PL_max_1 = zeros(num_locations, num_samples);
PL_max_2 = zeros(num_locations, num_samples);
PL_max_3 = zeros(num_locations, num_samples); 

pooled_1 = cell(num_locations * num_samples, 1);
pooled_2 = cell(num_locations * num_samples, 1);
pooled_3 = cell(num_locations * num_samples, 1);



%% Calculate the privacy loss
% Iterate over all locations (A to T)
for i = 1:num_locations   

    A1 = original_distance_matrices{i};
    B1 = GF1{i};
    per_sample_fractions_1 = zeros(num_samples, 1);  % For storing per-sample %

    for sample_idx = 1:num_samples
        % Add noise to the distance matrix to the location i to generate a new noisy sample
        % EPSILON = sample_idx; 
        noisy_distance_sample_1 = add_noise_to_distance_matrix(F1{i}, EPSILON);  
        noisy_sample_1 = add_noise_to_distance_matrix(B1, EPSILON);  

        min_rows_1 = min(size(A1,1), size(noisy_sample_1,1));
        min_cols_1 = min(size(A1,2), size(noisy_sample_1,2));
        A1_trimmed = A1(1:min_rows_1, 1:min_cols_1);
        noisy_trimmed_1 = noisy_sample_1(1:min_rows_1, 1:min_cols_1);       

        total_entries_1 = min_rows_1 * min_cols_1;

        violation_mask_1 = noisy_trimmed_1 > A1_trimmed;
        num_violations_1 = sum(violation_mask_1(:));                              % Count where noisy_sample > A3

        per_sample_fractions_1(sample_idx) = num_violations_1 / total_entries_1;    % Fraction of violations in this sample

        num_violations_per_location_1(i) = num_violations_per_location_1(i) + num_violations_1;
        total_entries_per_location_1(i) = total_entries_per_location_1(i) + total_entries_1;

        total_violations_all_1 = total_violations_all_1 + num_violations_1;         % Accumulate global counters
        total_entries_all_1 = total_entries_all_1 + total_entries_1;
        
        
        % Compute the posterior probability for this noisy distance sample
        sample_idx_ = (i-1)*num_samples + sample_idx; 
        [posterior_prob_sample_1{sample_idx_}, PL_matrix_1{sample_idx_}, PL_max_1(i, sample_idx)] = compute_log_posterior(F1, noisy_distance_sample_1, distance_matrix, num_locations, B_VALUE, CARDINALITY_N); 
        pooled_1{sample_idx_} = single(PL_matrix_1{sample_idx_}(:));
        PL_matrix_1{sample_idx_} = []; 
    end



    A2 = obfuscated_distance_matrices{i};   
    B2 = GF2{i};
    per_sample_fractions_2 = zeros(num_samples, 1);  % For storing per-sample %
    
    for sample_idx = 1:num_samples
        % Add noise to the distance matrix to the location i to generate a new noisy sample
        % EPSILON = sample_idx; 
        noisy_distance_sample_2 = add_noise_to_distance_matrix(F2{i}, EPSILON);  
        noisy_sample_2 = add_noise_to_distance_matrix(B2, EPSILON);  

        min_rows_2 = min(size(A2,1), size(noisy_sample_2,1));
        min_cols_2 = min(size(A2,2), size(noisy_sample_2,2));
        A2_trimmed = A2(1:min_rows_2, 1:min_cols_2);
        noisy_trimmed_2 = noisy_sample_2(1:min_rows_2, 1:min_cols_2);       

        total_entries_2 = min_rows_2 * min_cols_2;

        violation_mask_2 = noisy_trimmed_2 > A2_trimmed;
        num_violations_2 = sum(violation_mask_2(:));                              % Count where noisy_sample > A3

        per_sample_fractions_2(sample_idx) = num_violations_2 / total_entries_2;    % Fraction of violations in this sample

        num_violations_per_location_2(i) = num_violations_per_location_2(i) + num_violations_2;
        total_entries_per_location_2(i) = total_entries_per_location_2(i) + total_entries_2;

        total_violations_all_2 = total_violations_all_2 + num_violations_2;         % Accumulate global counters
        total_entries_all_2 = total_entries_all_2 + total_entries_2;

        
        % Compute the posterior probability for this noisy distance sample
        sample_idx__ = (i-1)*num_samples + sample_idx; 
        [posterior_prob_sample_2{sample_idx__}, PL_matrix_2{sample_idx__}, PL_max_2(i, sample_idx)] = compute_log_posterior(F2, noisy_distance_sample_2, distance_matrix, num_locations, B_VALUE, CARDINALITY_N);    
        pooled_2{sample_idx__} = single(PL_matrix_2{sample_idx__}(:));
        PL_matrix_2{sample_idx__} = [];    
    end


    for sample_idx = 1:num_samples
        % Add noise to the distance matrix to the location i to generate a new noisy sample
        % EPSILON = sample_idx; 
        noisy_distance_sample = add_noise_to_distance_matrix(F3{i}, EPSILON);  

        % Compute the posterior probability for this noisy distance sample
        sample_idx___ = (i-1)*num_samples + sample_idx; 
        [posterior_prob_sample_3{sample_idx___}, PL_matrix_3{sample_idx___}, PL_max_3(i, sample_idx)] = compute_log_posterior(F3, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N); 
        pooled_3{sample_idx___} = single(PL_matrix_3{sample_idx___}(:));
        PL_matrix_3{sample_idx___} = [];  
    end


    
    % Store average % for the location
    percentage_violations_1(i) = mean(per_sample_fractions_1) * 100;
    violation_fractions_1{i} = per_sample_fractions_1;
    fprintf('Location %d: %.2f%% of entries (on average) where noisy > original over %d samples.\n', ...
            i, percentage_violations_1(i), num_samples);
    
    disp('Posterior Probability 1 Sample:');
    disp(posterior_prob_sample_1{sample_idx_});

    % Store the privacy losses for this location
    % privacy_loss(i, :) = privacy_loss_samples';

    % % Print all privacy_losses for this location
    % fprintf('Privacy Loss for Location %d:\n', i);
    % disp(privacy_loss_samples);
    
    % Compute the supremum (max) of the absolute log of privacy losses for this location
    privacy_loss_supremum_1(i) = max(PL_max_1(i, :));
    fprintf('Supremum of abs(log(Privacy Loss 1)) for Location %d = %.6f\n\n', i, privacy_loss_supremum_1(i));




    % Store average % for the location
    percentage_violations_2(i) = mean(per_sample_fractions_2) * 100;
    violation_fractions_2{i} = per_sample_fractions_2;
    fprintf('Location %d: %.2f%% of entries (on average) where noisy > original over %d samples.\n', ...
            i, percentage_violations_2(i), num_samples);

    disp('Posterior Probability 2 Sample:');
    disp(posterior_prob_sample_2{sample_idx__});
    
    % Compute the supremum (max) of the absolute log of privacy losses for this location
    privacy_loss_supremum_2(i) = max(PL_max_2(i, :));
    fprintf('Supremum of abs(log(Privacy Loss 2)) for Location %d = %.6f\n\n', i, privacy_loss_supremum_2(i));




    disp('Posterior Probability 3 Sample:');
    disp(posterior_prob_sample_3{sample_idx___});
    
    % Compute the supremum (max) of the absolute log of privacy losses for this location
    privacy_loss_supremum_3(i) = max(PL_max_3(i, :));
    fprintf('Supremum of abs(log(Privacy Loss 3)) for Location %d = %.6f\n\n', i, privacy_loss_supremum_3(i));
end



% % Compute the average privacy loss supremum over all locations
% if isscalar(privacy_loss_supremum)
%     average_privacy_loss = privacy_loss_supremum;
% else
pooled_1_vec = vertcat(pooled_1{:});
average_privacy_loss_1 = mean(privacy_loss_supremum_1);
% end
fprintf('Average Supremum of abs(log(Privacy Loss 1)) over all locations: %.6f\n', average_privacy_loss_1);


pooled_2_vec = vertcat(pooled_2{:});
average_privacy_loss_2 = mean(privacy_loss_supremum_2);
fprintf('Average Supremum of abs(log(Privacy Loss 2)) over all locations: %.6f\n', average_privacy_loss_2);


pooled_3_vec = vertcat(pooled_3{:});
average_privacy_loss_3 = mean(privacy_loss_supremum_3);
fprintf('Average Supremum of abs(log(Privacy Loss 3)) over all locations: %.6f\n\n', average_privacy_loss_3);


pause(11);
time_PL = toc;





% fprintf('\n Number of Violating Entries 1 per Location:\n');
% for i = 1:num_locations
%     fprintf('Location %2d: %d / %d entries violated (%.2f%%)\n', ...
%         i, num_violations_per_location_1(i), total_entries_per_location_1(i), ...
%         100 * num_violations_per_location_1(i) / total_entries_per_location_1(i));
% end

fprintf('\n Global Summary 1:\n');
fprintf('Total Violating Entries 1: %d\n', total_violations_all_1);
fprintf('Total Entries Evaluated 1: %d\n', total_entries_all_1);
fprintf('Global Violation Percentage 1: %.4f%%\n', ...
    100 * total_violations_all_1 / total_entries_all_1);
% ================================================================================================


% fprintf('\n Number of Violating Entries 2 per Location:\n');
% for i = 1:num_locations
%     fprintf('Location %2d: %d / %d entries violated (%.2f%%)\n', ...
%         i, num_violations_per_location_2(i), total_entries_per_location_2(i), ...
%         100 * num_violations_per_location_2(i) / total_entries_per_location_2(i));
% end

fprintf('\n Global Summary 2:\n');
fprintf('Total Violating Entries 2: %d\n', total_violations_all_2);
fprintf('Total Entries Evaluated 2: %d\n', total_entries_all_2);
fprintf('Global Violation Percentage 1: %.4f%%\n', ...
    100 * total_violations_all_2 / total_entries_all_2);
% ================================================================================================


for i = 1:num_locations
    fprintf('Location %d - Utility Loss (Cost vs B3): %.6f\n', i, utility_loss(i));    
end





fprintf('Data Loading time: %.2f seconds\n', time_data );
% fprintf('Select Target Nodes time: %.2f seconds\n', time_select_target_nodes );
fprintf('Info Loading time: %.2f seconds\n', time_info );
fprintf('Compute Distance Matrix time: %.2f seconds\n', time_compute_distance_matrix );
fprintf('Original Distance Matrix time: %.2f seconds\n', time_original_distance );
fprintf('Obfuscated Distance Matrix time: %.2f seconds\n', time_obfuscated_distance );
fprintf('Cost Original  Matrix time: %.2f seconds\n', time_cost_original_distance );
fprintf('Cost Obfuscated Matrix time: %.2f seconds\n', time_cost_obfuscated_distance );
fprintf('Cost Matrix time: %.2f seconds\n', time_cost_distance );
fprintf('Gaussian Fit time: %.2f seconds\n', time_gaussian_fit );
fprintf('Privacy Loss time: %.2f seconds\n', time_PL );

% computation_time  = toc;
% fprintf('Computation time: %.2f seconds\n', computation_time );





% PDF plot
figure; [f, xi] = ksdensity(pooled_1_vec);
plot(xi, f, 'LineWidth', 2);
xlabel('abs(log(Privacy Loss))');
ylabel('Density');
title('PDF of Privacy Loss 1 (all entries)');
grid on;

% PDF plot
figure; [f_2, xi_2] = ksdensity(pooled_2_vec);
plot(xi_2, f_2, 'LineWidth', 2);
xlabel('abs(log(Privacy Loss))');
ylabel('Density');
title('PDF of Privacy Loss 2 (all entries)');
grid on;

% PDF plot
figure; [f_3, xi_3] = ksdensity(pooled_3_vec);
plot(xi_3, f_3, 'LineWidth', 2);
xlabel('abs(log(Privacy Loss))');
ylabel('Density');
title('PDF of Privacy Loss 3 (all entries)');
grid on;




% % CDF plot
% figure;
% [f_cdf, xi_cdf] = ksdensity(pooled_1, 'Function', 'cdf');
% plot(xi_cdf, f_cdf, 'LineWidth', 2);
% xlabel('abs(log(Privacy Loss))');
% ylabel('Cumulative Probability');
% title('CDF of Privacy Loss 1');
% grid on;
% 
% % CDF plot
% figure;
% [f_cdf_2, xi_cdf_2] = ksdensity(pooled_2, 'Function', 'cdf');
% plot(xi_cdf_2, f_cdf_2, 'LineWidth', 2);
% xlabel('abs(log(Privacy Loss))');
% ylabel('Cumulative Probability');
% title('CDF of Privacy Loss 2');
% grid on;
% 
% % CDF plot
% figure;
% [f_cdf_3, xi_cdf_3] = ksdensity(pooled_3, 'Function', 'cdf');
% plot(xi_cdf_3, f_cdf_3, 'LineWidth', 2);
% xlabel('abs(log(Privacy Loss))');
% ylabel('Cumulative Probability');
% title('CDF of Privacy Loss 3');
% grid on;






avg_val = mean(PL_matrix_1{sample_idx_}(:));

% Build the CDF
figure;
[f_cdf, xi_cdf] = ksdensity(pooled_1_vec, 'Function', 'cdf');
plot(xi_cdf, f_cdf, 'LineWidth', 2);
hold on;

% Plot the original vertical/horizontal line for the mean
cdf_at_mean = interp1(xi_cdf, f_cdf, avg_val);
plot([avg_val, avg_val], [0, cdf_at_mean], '--r', 'LineWidth', 1.5);
plot([min(xi_cdf), avg_val], [cdf_at_mean, cdf_at_mean], '--r', 'LineWidth', 1.5);
scatter(avg_val, cdf_at_mean, 50, 'r', 'filled');
text(avg_val, cdf_at_mean, sprintf('  Mean = %.4f', avg_val), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

% Define CDF thresholds
cdf_levels = [0.8, 0.85, 0.9, 0.95, 0.99];
colors = ['r', 'r', 'r', 'r'];  % Same color; change if you want different ones

% Loop through each level and draw intersecting lines
for i = 1:length(cdf_levels)
    y_val = cdf_levels(i);
    [unique_f_cdf, ia, ~] = unique(f_cdf);
    unique_xi_cdf = xi_cdf(ia);
    x_val = interp1(unique_f_cdf, unique_xi_cdf, y_val, 'linear', 'extrap');      % Interpolate x from y
 
    % Vertical line from x-axis to the CDF curve
    plot([x_val, x_val], [0, y_val], '--r', 'LineWidth', 1.5);

    % Horizontal line from y-axis to the x point
    plot([min(xi_cdf), x_val], [y_val, y_val], '--r', 'LineWidth', 1.5);

    % Optional: add label at the intersection
    scatter(x_val, y_val, 50, 'r', 'filled');
    text(x_val, y_val, sprintf('  %.2f', y_val), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end

hold off;
xlabel('abs(log(Privacy Loss 1))');
ylabel('Cumulative Probability 1');
title('CDF of Privacy Loss 1 with Mean and Selected CDF Thresholds');
grid on;
% ===================================================================================




avg_val_2 = mean(PL_matrix_2{sample_idx__}(:));

% Build the CDF
figure;
[f_cdf_2, xi_cdf_2] = ksdensity(pooled_2_vec, 'Function', 'cdf');
plot(xi_cdf_2, f_cdf_2, 'LineWidth', 2);
hold on;

% Plot the original vertical/horizontal line for the mean
cdf_at_mean_2 = interp1(xi_cdf_2, f_cdf_2, avg_val_2);
plot([avg_val_2, avg_val_2], [0, cdf_at_mean_2], '--r', 'LineWidth', 1.5);
plot([min(xi_cdf_2), avg_val_2], [cdf_at_mean_2, cdf_at_mean_2], '--r', 'LineWidth', 1.5);
scatter(avg_val_2, cdf_at_mean_2, 50, 'r', 'filled');
text(avg_val_2, cdf_at_mean_2, sprintf('  Mean = %.4f', avg_val_2), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

% Define CDF thresholds
cdf_levels_2 = [0.8, 0.85, 0.9, 0.95, 0.99];
colors = ['r', 'r', 'r', 'r'];  % Same color; change if you want different ones

% Loop through each level and draw intersecting lines
for i = 1:length(cdf_levels_2)
    y_val_2 = cdf_levels_2(i);
    [unique_f_cdf_2, ia_2, ~] = unique(f_cdf_2);
    unique_xi_cdf_2 = xi_cdf_2(ia_2);
    x_val_2 = interp1(unique_f_cdf_2, unique_xi_cdf_2, y_val_2, 'linear', 'extrap');      % Interpolate x from y
 
    % Vertical line from x-axis to the CDF curve
    plot([x_val_2, x_val_2], [0, y_val_2], '--r', 'LineWidth', 1.5);

    % Horizontal line from y-axis to the x point
    plot([min(xi_cdf_2), x_val_2], [y_val_2, y_val_2], '--r', 'LineWidth', 1.5);

    % Optional: add label at the intersection
    scatter(x_val_2, y_val_2, 50, 'r', 'filled');
    text(x_val_2, y_val_2, sprintf('  %.2f', y_val_2), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end

hold off;
xlabel('abs(log(Privacy Loss 2))');
ylabel('Cumulative Probability 2');
title('CDF of Privacy Loss 2 with Mean and Selected CDF Thresholds');
grid on;
% ===================================================================================




avg_val_3 = mean(PL_matrix_3{sample_idx___}(:));

% Build the CDF
figure;
[f_cdf_3, xi_cdf_3] = ksdensity(pooled_3_vec, 'Function', 'cdf');
plot(xi_cdf_3, f_cdf_3, 'LineWidth', 2);
hold on;

% Plot the original vertical/horizontal line for the mean
cdf_at_mean_3 = interp1(xi_cdf_3, f_cdf_3, avg_val_3);
plot([avg_val_3, avg_val_3], [0, cdf_at_mean_3], '--r', 'LineWidth', 1.5);
plot([min(xi_cdf_3), avg_val_3], [cdf_at_mean_3, cdf_at_mean_3], '--r', 'LineWidth', 1.5);
scatter(avg_val_3, cdf_at_mean_3, 50, 'r', 'filled');
text(avg_val_3, cdf_at_mean_3, sprintf('  Mean = %.4f', avg_val_3), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

% Define CDF thresholds
cdf_levels_3 = [0.8, 0.85, 0.9, 0.95, 0.99];
colors = ['r', 'r', 'r', 'r'];  % Same color; change if you want different ones

% Loop through each level and draw intersecting lines
for i = 1:length(cdf_levels_3)
    y_val_3 = cdf_levels_3(i);
    [unique_f_cdf_3, ia_3, ~] = unique(f_cdf_3);
    unique_xi_cdf_3 = xi_cdf_3(ia_3);
    x_val_3 = interp1(unique_f_cdf_3, unique_xi_cdf_3, y_val_3, 'linear', 'extrap');      % Interpolate x from y
 
    % Vertical line from x-axis to the CDF curve
    plot([x_val_3, x_val_3], [0, y_val_3], '--r', 'LineWidth', 1.5);

    % Horizontal line from y-axis to the x point
    plot([min(xi_cdf_3), x_val_3], [y_val_3, y_val_3], '--r', 'LineWidth', 1.5);

    % Optional: add label at the intersection
    scatter(x_val_3, y_val_3, 50, 'r', 'filled');
    text(x_val_3, y_val_3, sprintf('  %.2f', y_val_3), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
end

hold off;
xlabel('abs(log(Privacy Loss 3))');
ylabel('Cumulative Probability 3');
title('CDF of Privacy Loss 3 with Mean and Selected CDF Thresholds');
grid on;



