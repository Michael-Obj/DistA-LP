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
TARGET_LON_MAX = 12.6; 
TARGET_LON_MIN = 12.401; 
TARGET_LAT_MAX = 42.1; 
TARGET_LAT_MIN = 41.901;

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


tic;   
% Call function to find nodes within the target region
[col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN);

pause(2);
time_select_target_nodes = toc;


tic;   
% Generate obfuscated locations as the first 40% of the selected points
num_obfuscated = max(1, floor(0.1 * length(original_longitude))); % Ensure at least one point is selected
obfuscated_longitude = original_longitude(1:num_obfuscated);
obfuscated_latitude = original_latitude(1:num_obfuscated);

target_lat = col_latitude_selected(1);        % Target latitude
target_long = col_longitude_selected(1);      % Target longitude



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
plot(original_longitude, original_latitude, 'ro', 'MarkerFaceColor', 'r'); % Highlight original nodes
plot(obfuscated_longitude, obfuscated_latitude, 'bs', 'MarkerFaceColor', 'g'); % Highlight obfuscated nodes
plot(target_long, target_lat, 'bs', 'MarkerFaceColor', 'y'); % Highlight target node
xlabel('Longitude');
ylabel('Latitude');
title('Selected, Original & Obfuscated Nodes & Target Nodes in Target Region');
grid on;
hold off;



% Compute pairwise Haversine distances and build raw and noisy distance matrices for each location
num_locations = length(original_longitude);  % Number of locations
num_obf = length(obfuscated_longitude);  % Number of obfuscated locations

original_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

obfuscated_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

cost_distance_matrix_original = cell(num_locations, 1);  % Cell array to store the 10x10 matrices
% cost_distance_matrix_obfuscated = cell(num_obf, 1); 
cost_coefficient_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices

% coeffs_F1 = cell(num_locations, 1);
% coeffs_F2 = cell(num_locations, 1);
% coeffs_F3 = cell(num_locations, 1);

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
EPSILON = 2.0;  % EPSILON value for noise
B_VALUE = 1/ EPSILON;  % B_VALUE for perturbation probability calculation
CARDINALITY_N = 10;
lambda2 = 1.0;
lambda3 = 1.0;






tic;   
%% Loop through all locations (A, B, C, ..., T) and create their original raw distance
for i = 1:num_locations
    [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);
    
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
    [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);


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
    [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);
    
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

    cost_coefficient = zeros(CARDINALITY_N, num_obf);

    % Compute the cost coefficient matrix element-wise. Loop over each element in the original vector (neighbors) and each obfuscated distance.
    for i = 1:CARDINALITY_N
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


% deg = 3;                               % or 2, 4, …
% P1 = cell(num_locations,1);                  
% P2 = cell(num_locations,1);
% P3 = cell(num_locations,1);
% 
% for i = 1:num_locations
%     A1 = original_distance_matrices{i};
%     A2 = obfuscated_distance_matrices{i};
%     A3 = cost_coefficient_matrices{i};
% 
%     % Fit 
%     coeffs_F1{i} = fitPoly2D(A1, deg);
%     coeffs_F2{i} = fitPoly2D(A2, deg);
%     coeffs_F3{i} = fitPoly2D(A3, deg);
% 
%     P1{i} = evalPoly2D(coeffs_F1{i}, size(A1,1), size(A1,2), deg);
%     P2{i} = evalPoly2D(coeffs_F2{i}, size(A2,1), size(A2,2), deg);
%     P3{i} = evalPoly2D(coeffs_F3{i}, size(A3,1), size(A3,2), deg);
% end



tic;   
deg = 3;                               % or 2, 4, …
% preallocate
best_pi      = cell(num_locations,1);
best_coeffs  = cell(num_locations,1);

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
    [best_pi, best_coeffs{i}, GF1{i}, GF2{i}, GF3{i}] = reorder_fit_polynomials(A1, A2, A3, deg, 1.0, 1.0);

    F1_norm{i} = best_coeffs{i}(1, 1:10); 
    F2_norm{i} = best_coeffs{i}(1, 11:20); 
    F3_norm{i} = best_coeffs{i}(1, 21:30); 


    % Utility loss
    B3 = GF3{i};

    min_rows = min(size(A3,1), size(B3,1));
    min_cols = min(size(A3,2), size(B3,2));    
    A3_trimmed = A3(1:min_rows, 1:min_cols);
    B3_trimmed = B3(1:min_rows, 1:min_cols);

    utility_loss(i) = norm(A3_trimmed - B3_trimmed, 'fro');
end



M1 = vertcat(F1_norm{:});   % size is [num_locations × 10]
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

pooled_1 = [];
pooled_2 = [];
pooled_3 = [];



%% Calculate the posterior leakage
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
        pooled_1 = [pooled_1; PL_matrix_1{sample_idx_}(:)];
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
        pooled_2 = [pooled_2; PL_matrix_2{sample_idx__}(:)];
    end


    for sample_idx = 1:num_samples
        % Add noise to the distance matrix to the location i to generate a new noisy sample
        % EPSILON = sample_idx; 
        noisy_distance_sample = add_noise_to_distance_matrix(F3{i}, EPSILON);  

        % Compute the posterior probability for this noisy distance sample
        sample_idx___ = (i-1)*num_samples + sample_idx; 
        [posterior_prob_sample_3{sample_idx___}, PL_matrix_3{sample_idx___}, PL_max_3(i, sample_idx)] = compute_log_posterior(F3, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N); 
        pooled_3 = [pooled_3; PL_matrix_3{sample_idx___}(:)];
    end


    
    % Store average % for the location
    percentage_violations_1(i) = mean(per_sample_fractions_1) * 100;
    violation_fractions_1{i} = per_sample_fractions_1;
    fprintf('Location %d: %.2f%% of entries (on average) where noisy > original over %d samples.\n', ...
            i, percentage_violations_1(i), num_samples);
    
    disp('Posterior Probability 1 Sample:');
    disp(posterior_prob_sample_1{sample_idx_});

    % Store the posterior leakages for this location
    % posterior_leakages(i, :) = posterior_leakage_samples';

    % % Print all posterior leakages for this location
    % fprintf('Posterior Leakages for Location %d:\n', i);
    % disp(posterior_leakage_samples);
    
    % Compute the supremum (max) of the absolute log of posterior leakages for this location
    posterior_leakages_supremum_1(i) = max(PL_max_1(i, :));
    fprintf('Supremum of abs(log(Posterior Leakage 1)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum_1(i));




    % Store average % for the location
    percentage_violations_2(i) = mean(per_sample_fractions_2) * 100;
    violation_fractions_2{i} = per_sample_fractions_2;
    fprintf('Location %d: %.2f%% of entries (on average) where noisy > original over %d samples.\n', ...
            i, percentage_violations_2(i), num_samples);

    disp('Posterior Probability 2 Sample:');
    disp(posterior_prob_sample_2{sample_idx__});
    
    % Compute the supremum (max) of the absolute log of posterior leakages for this location
    posterior_leakages_supremum_2(i) = max(PL_max_2(i, :));
    fprintf('Supremum of abs(log(Posterior Leakage 2)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum_2(i));




    disp('Posterior Probability 3 Sample:');
    disp(posterior_prob_sample_3{sample_idx___});
    
    % Compute the supremum (max) of the absolute log of posterior leakages for this location
    posterior_leakages_supremum_3(i) = max(PL_max_3(i, :));
    fprintf('Supremum of abs(log(Posterior Leakage 3)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum_3(i));
end



% % Compute the average posterior leakage supremum over all locations
% if isscalar(posterior_leakages_supremum)
%     average_posterior_leakage = posterior_leakages_supremum;
% else
average_posterior_leakage_1 = mean(posterior_leakages_supremum_1);
% end
fprintf('Average Supremum of abs(log(Posterior Leakage 1)) over all locations: %.6f\n', average_posterior_leakage_1);


average_posterior_leakage_2 = mean(posterior_leakages_supremum_2);
fprintf('Average Supremum of abs(log(Posterior Leakage 2)) over all locations: %.6f\n', average_posterior_leakage_2);


average_posterior_leakage_3 = mean(posterior_leakages_supremum_3);
fprintf('Average Supremum of abs(log(Posterior Leakage 3)) over all locations: %.6f\n\n', average_posterior_leakage_3);


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
fprintf('Select Target Nodes time: %.2f seconds\n', time_select_target_nodes );
fprintf('Info Loading time: %.2f seconds\n', time_info );
fprintf('Compute Distance Matrix time: %.2f seconds\n', time_compute_distance_matrix );
fprintf('Original Distance Matrix time: %.2f seconds\n', time_original_distance );
fprintf('Obfuscated Distance Matrix time: %.2f seconds\n', time_obfuscated_distance );
fprintf('Cost Original  Matrix time: %.2f seconds\n', time_cost_original_distance );
fprintf('Cost Obfuscated Matrix time: %.2f seconds\n', time_cost_obfuscated_distance );
fprintf('Cost Matrix time: %.2f seconds\n', time_cost_distance );
fprintf('Gaussian Fit time: %.2f seconds\n', time_gaussian_fit );
fprintf('Posterior Leakage time: %.2f seconds\n', time_PL );

% computation_time  = toc;
% fprintf('Computation time: %.2f seconds\n', computation_time );





% PDF plot
figure; [f, xi] = ksdensity(pooled_1);
plot(xi, f, 'LineWidth', 2);
xlabel('abs(log(Posterior Leakage))');
ylabel('Density');
title('PDF of Posterior Leakage 1 (all entries)');
grid on;

% PDF plot
figure; [f_2, xi_2] = ksdensity(pooled_2);
plot(xi_2, f_2, 'LineWidth', 2);
xlabel('abs(log(Posterior Leakage))');
ylabel('Density');
title('PDF of Posterior Leakage 2 (all entries)');
grid on;

% PDF plot
figure; [f_3, xi_3] = ksdensity(pooled_3);
plot(xi_3, f_3, 'LineWidth', 2);
xlabel('abs(log(Posterior Leakage))');
ylabel('Density');
title('PDF of Posterior Leakage 3 (all entries)');
grid on;




% % CDF plot
% figure;
% [f_cdf, xi_cdf] = ksdensity(pooled_1, 'Function', 'cdf');
% plot(xi_cdf, f_cdf, 'LineWidth', 2);
% xlabel('abs(log(Posterior Leakage))');
% ylabel('Cumulative Probability');
% title('CDF of Posterior Leakage 1');
% grid on;
% 
% % CDF plot
% figure;
% [f_cdf_2, xi_cdf_2] = ksdensity(pooled_2, 'Function', 'cdf');
% plot(xi_cdf_2, f_cdf_2, 'LineWidth', 2);
% xlabel('abs(log(Posterior Leakage))');
% ylabel('Cumulative Probability');
% title('CDF of Posterior Leakage 2');
% grid on;
% 
% % CDF plot
% figure;
% [f_cdf_3, xi_cdf_3] = ksdensity(pooled_3, 'Function', 'cdf');
% plot(xi_cdf_3, f_cdf_3, 'LineWidth', 2);
% xlabel('abs(log(Posterior Leakage))');
% ylabel('Cumulative Probability');
% title('CDF of Posterior Leakage 3');
% grid on;






avg_val = mean(PL_matrix_1{sample_idx_}(:));

% Build the CDF
figure;
[f_cdf, xi_cdf] = ksdensity(pooled_1, 'Function', 'cdf');
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
xlabel('abs(log(Posterior Leakage 1))');
ylabel('Cumulative Probability 1');
title('CDF of Posterior Leakage 1 with Mean and Selected CDF Thresholds');
grid on;
% ===================================================================================




avg_val_2 = mean(PL_matrix_2{sample_idx__}(:));

% Build the CDF
figure;
[f_cdf_2, xi_cdf_2] = ksdensity(pooled_2, 'Function', 'cdf');
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
xlabel('abs(log(Posterior Leakage 2))');
ylabel('Cumulative Probability 2');
title('CDF of Posterior Leakage 2 with Mean and Selected CDF Thresholds');
grid on;
% ===================================================================================




avg_val_3 = mean(PL_matrix_3{sample_idx___}(:));

% Build the CDF
figure;
[f_cdf_3, xi_cdf_3] = ksdensity(pooled_3, 'Function', 'cdf');
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
xlabel('abs(log(Posterior Leakage 3))');
ylabel('Cumulative Probability 3');
title('CDF of Posterior Leakage 3 with Mean and Selected CDF Thresholds');
grid on;



