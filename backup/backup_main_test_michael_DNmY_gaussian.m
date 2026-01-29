addpath('./functions/');
% parameters; 


%% Load the map dataset
opts = detectImportOptions('./datasets/rome/rome_nodes.csv');
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable('./datasets/rome/rome_nodes.csv', opts);
df_edges = readtable('./datasets/rome/rome_edges.csv');


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
% TARGET_LON_MAX = 12.3; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 42.01;

% ----------------------------
% ----------------------------
% TARGET REGIONS (10)
% ----------------------------
% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 42.01;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.801;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 41.8; 
% TARGET_LAT_MIN = 41.701;
% ----------------------------
% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 42.01;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.801;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 41.8; 
% TARGET_LAT_MIN = 41.701;
% ----------------------------
% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.601; 
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.601; 
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.801;
% ----------------------------
% ----------------------------

TARGET_LON_MAX = 12.8; 
TARGET_LON_MIN = 12.2; 
TARGET_LAT_MAX = 42.1; 
TARGET_LAT_MIN = 41.65;


% Call function to find nodes within the target region
[col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN);




% Generate obfuscated locations as the first 40% of the selected points
num_obfuscated = max(1, floor(0.1 * length(original_longitude))); % Ensure at least one point is selected
obfuscated_longitude = original_longitude(1:num_obfuscated);
obfuscated_latitude = original_latitude(1:num_obfuscated);




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








% The following plot is just for testing 
figure;
plot(col_longitude, col_latitude, 'o'); 
hold on; 
plot(col_longitude_selected, col_latitude_selected, 'bs', 'MarkerFaceColor', 'b'); % Highlight selected nodes
plot(original_longitude, original_latitude, 'ro', 'MarkerFaceColor', 'r'); % Highlight original nodes
plot(obfuscated_longitude, obfuscated_latitude, 'bs', 'MarkerFaceColor', 'g'); % Highlight obfuscated nodes
xlabel('Longitude');
ylabel('Latitude');
title('Selected, Original & Obfuscated Nodes in Target Region');
grid on;
hold off;


% Compute pairwise Haversine distances and build raw and noisy distance matrices for each location
num_locations = length(original_longitude);  % Number of locations
raw_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices
quantized_matrices = cell(num_locations, 1);
noisy_distance_matrices = cell(num_locations, 1); % Cell array to store the noisy 10x10 matrices
perturbation_probabilities = cell(num_locations, 1); % Cell array to store perturbation probabilities
posterior_probabilities = cell(num_locations, 1); % Cell array to store posterior probabilities


distance_matrix = compute_distance_matrix(original_latitude, original_longitude); 
% raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude);


%% Use captial letters for the constants 
EPSILON = 2;  % EPSILON value for noise
B_VALUE = 1/ EPSILON;  % B_VALUE for perturbation probability calculation
CARDINALITY_N = 5;




%% Loop through all locations (A, B, C, ..., T) and create their raw distance/cost matrices
for i = 1:num_locations
    % Select the 9 nearest neighbors for the current location
    [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, CARDINALITY_N - 1);
    
    % % Compute the raw distance matrix for this location and its 9 nearest neighbors
    % raw_distance_matrix = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], [original_latitude(i); nearest_latitude], ...
    %                                                    [obfuscated_longitude; nearest_longitude], [obfuscated_latitude; nearest_latitude]);

    % % Compute the raw distance matrix before alignment
    % raw_distance_matrix_unaligned = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], ...
    %                                                             [original_latitude(i); nearest_latitude], ...
    %                                                             [obfuscated_longitude; nearest_longitude], ...
    %                                                             [obfuscated_latitude; nearest_latitude]);

    % Compute the raw distance matrix before alignment
    raw_distance_matrix_unaligned = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], ...
                                                                [original_latitude(i); nearest_latitude], ...
                                                                obfuscated_longitude, ...
                                                                obfuscated_latitude);

    
    % Store the aligned distance matrix in the cell array
    raw_distance_matrices{i} = raw_distance_matrix_unaligned;

    
    % % Store the distance matrix in the cell array
    % raw_distance_matrices{i} = raw_distance_matrix;


    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf('Raw Distance Matrix:\n');
    disp(raw_distance_matrices{i});

    % % Compute success probability for raw_distance_matrix
    % sp_raw = compute_success_probability(raw_distance_matrix_unaligned, raw_distance_matrix);
    % fprintf('Success Probability (raw): %.2f%%\n', sp_raw * 100);
end






for i = 1:num_locations
    % Step 2: Compute the Distance Matrix
    % Compute pairwise Euclidean distance matrix
    D = raw_distance_matrices{i};
    
    % Generate utiltiy loss matrix
    UL = rand(num_locations, 20); 
    
    
    % % Step 3: Permute the Distance Matrix Without Fitting
    % % Generate a random permutation of indices
    % perm_indices = randperm(N);
    % 
    % % Apply the permutation to both rows and columns to maintain symmetry
    % D_permuted = D(perm_indices, perm_indices);
    
    
    % Step 4: Fit a 2D Gaussian to the Original Distance Matrix
    max_iters = 2000; % Number of iterations for the fitting algorithm
    
    
    % Fit the Gaussian to the original distance matrix
    [G_fit, params, pi_opt] = fit_gaussian_to_matrix(D, max_iters);
    
    
    % Permute the original distance matrix according to the optimal permutation
    D_fitted_permuted = D(pi_opt, pi_opt);

    quantized_matrices{i} = D_fitted_permuted;    
    
    
    
    % [P_real, expected_loss_real] = data_perturbation(UL(pi_opt, :), D_fitted_permuted, 1); 
    % 
    % 
    % [P_fit, expected_loss_fit] = data_perturbation(UL(pi_opt, :), G_fit, 1); 
    
    
    
    
    % Visualization
    figure;
    
    
    subplot(1, 3, 1);
    imagesc(D);
    colorbar;
    title('Randomly Permuted Distance Matrix');
    
    
    subplot(1, 3, 2);
    imagesc(D_fitted_permuted);
    colorbar;
    title('Fitted Permuted Distance Matrix');
    
    
    subplot(1, 3, 3);
    imagesc(G_fit);
    colorbar;
    title('Fitted Gaussian Matrix');

end










% Number of locations (A, B, C, ..., T)
% num_locations = 20;
num_samples = 100; % Number of samples for each posterior probability
posterior_leakages = zeros(num_locations, num_samples);  % To store posterior leakages for each location




%% Calculate the posterior leakage
% Iterate over all locations (A to T)
for i = 1:num_locations    
    for sample_idx = 1:num_samples
        % Add noise to the distance matrix to the location i to generate a new noisy sample
        % EPSILON = sample_idx; 
        noisy_distance_sample = add_noise_to_distance_matrix(quantized_matrices{i}, EPSILON, 'laplace');  % noiseType: 'laplace' or 'gaussian'
        
        % Compute the posterior probability for this noisy distance sample
        sample_idx_ = (i-1)*num_locations + sample_idx; 
        [posterior_prob_sample{sample_idx_}, PL_matrix{sample_idx_}, PL_max(i, sample_idx)] = compute_log_posterior(quantized_matrices, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N);        
    end

    disp('Posterior Probability Sample:');
    disp(posterior_prob_sample{sample_idx_});
   
    % Store the posterior leakages for this location
    % posterior_leakages(i, :) = posterior_leakage_samples';

    % % Print all posterior leakages for this location
    % fprintf('Posterior Leakages for Location %d:\n', i);
    % disp(posterior_leakage_samples);
    
    % Compute the supremum (max) of the absolute log of posterior leakages for this location
    posterior_leakages_supremum = max(PL_max(i, :));
    fprintf('Supremum of abs(log(Posterior Leakage)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum);
end




% % Compute the average posterior leakage supremum over all locations
% if isscalar(posterior_leakages_supremum)
%     average_posterior_leakage = posterior_leakages_supremum;
% else
average_posterior_leakage = mean(posterior_leakages_supremum);
% end
fprintf('Average Supremum of abs(log(Posterior Leakage)) over all locations: %.6f\n', average_posterior_leakage);