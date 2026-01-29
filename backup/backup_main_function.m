addpath('./functions/');
parameters; 



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
fprintf("Longitude Range: [%.6f, %.6f]\n", min(col_longitude), max(col_longitude));
fprintf("Latitude Range:  [%.6f, %.6f]\n", min(col_latitude), max(col_latitude));

% Define target region bounds 
TARGET_LON_MAX = 12.3; 
TARGET_LON_MIN = 12.2; 
TARGET_LAT_MAX = 42.1; 
TARGET_LAT_MIN = 42.01;

% Call function to find nodes within the target region
[col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN);


% Generate obfuscated locations as the first 40% of the selected points
num_obfuscated = max(1, floor(0.4 * length(original_longitude))); % Ensure at least one point is selected
obfuscated_longitude = original_longitude(1:num_obfuscated);
obfuscated_latitude = original_latitude(1:num_obfuscated);


% Debug: Print the number of original and obfuscated nodes
fprintf("Number of selected nodes: %d\n", length(col_osmid_selected));
disp('Col_Osmid_Selected:');
disp(col_osmid_selected);


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
noisy_distance_matrices = cell(num_locations, 1); % Cell array to store the noisy 10x10 matrices
perturbation_probabilities = cell(num_locations, 1); % Cell array to store perturbation probabilities
posterior_probabilities = cell(num_locations, 1); % Cell array to store posterior probabilities

distance_matrix = compute_distance_matrix(original_latitude, original_longitude); 


%% Use captial letters for the constants 
epsilon = 1.0;  % Epsilon value for noise
b_value = 0.5;  % b value for perturbation probability calculation
cardinality_N = 10;


%% Loop through all locations (A, B, C, ..., T) and create their raw distance/cost matrices
for i = 1:num_locations
    % Select the 9 nearest neighbors for the current location
    [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, 9);
    
    % Compute the raw distance matrix for this location and its 9 nearest neighbors
    raw_distance_matrix = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], [original_latitude(i); nearest_latitude], ...
                                                       [obfuscated_longitude; nearest_longitude], [obfuscated_latitude; nearest_latitude]);
    
    % Store the distance matrix in the cell array
    raw_distance_matrices{i} = raw_distance_matrix;

    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf('Raw Distance Matrix:\n');
    disp(raw_distance_matrices{i});
end


% % Compute the noisy distance matrix for location 1 (D_i)
% noisy_distance_matrix = add_noise_to_distance_matrix(raw_distance_matrices{1}, epsilon);
% % noisy_distance_matrices{i} = noisy_distance_matrix;
% fprintf('Noisy Distance Matrix:\n');
% disp(noisy_distance_matrix);

% for i = 1:num_locations    
%     % Compute the perturbation probability P(D_i|i)
%     perturbation_prob = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_matrix, b_value, cardinality_N);
%     % Store perturbation probability for location i
%     perturbation_probabilities{i} = perturbation_prob;
% 
%     % Display Results
%     fprintf('Results for Location %d:\n', i);
%     fprintf('Perturbation Probability Matrix:\n');
%     disp(perturbation_probabilities{i});
% end

%% Qiu: I don't think we need this part anymore
% for i = 1:num_locations
%     % Compute the posterior probability P(i|D_i)
%     posterior_prob = compute_log_posterior(raw_distance_matrices, noisy_distance_matrix, num_locations, b_value, cardinality_N);
%     % posterior_prob = compute_log_posterior(perturbation_probabilities{i}, num_locations);
%     % Store posterior probability for location i
%     posterior_probabilities{i} = posterior_prob;
% 
%     % Display Results
%     fprintf('Results for Location %d:\n', i);
%     fprintf('Posterior Probability Matrix:\n');
%     disp(posterior_probabilities{i});
% end


% Number of locations (A, B, C, ..., T)
num_locations = 20;
num_samples = 100; % Number of samples for each posterior probability
posterior_leakages = zeros(num_locations, num_samples);  % To store posterior leakages for each location


% Iterate over all locations (A to T)
for i = 1:num_locations
    % % % Compute the noisy distance matrix for the current location (D_i)
    % % noisy_distance_matrix = add_noise_to_distance_matrix(raw_distance_matrices{i}, epsilon);
    % 
    % % Compute the perturbation probability P(D_i|i)
    % perturbation_prob = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_matrix, b_value, cardinality_N);
    % 
    % % Compute the posterior probability P(i|D_i) for this location
    % posterior_probabilities = compute_log_posterior(raw_distance_matrices{i}, noisy_distance_matrix, num_locations, b_value, cardinality_N);
    % 
    % % Sample the noisy distance matrix 100 times for posterior leakage computation
    % posterior_leakage_samples = zeros(num_samples, 1);
    
    for sample_idx = 1:num_samples
        % epsilon = sample_idx;


        % Add noise to the distance matrix to the location i to generate a new noisy sample
        noisy_distance_sample = add_noise_to_distance_matrix(raw_distance_matrices{i}, epsilon);
        
        % Compute perturbation probability for this noisy distance sample
        perturbation_prob_sample = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_sample, b_value, cardinality_N);
        
        % Compute the posterior probability for this noisy distance sample
        sample_idx_ = (i-1)*num_locations + sample_idx; 
        [posterior_prob_sample{sample_idx_}, PL_matrix{sample_idx_}, PL_max(sample_idx_)] = compute_log_posterior(raw_distance_matrices, noisy_distance_sample, distance_matrix, num_locations, b_value, cardinality_N);
        
        % Compute the posterior leakage for this sample
        % posterior_leakage_samples(sample_idx) = compute_posterior_leakage_value(posterior_prob_sample, posterior_probabilities);
    end
   
    % Store the posterior leakages for this location
    % posterior_leakages(i, :) = posterior_leakage_samples';


    % % Print all posterior leakages for this location
    % fprintf('Posterior Leakages for Location %d:\n', i);
    % disp(posterior_leakage_samples);
    
    % Compute the supremum (max) of the absolute log of posterior leakages for this location
    % posterior_leakages_supremum = max(abs(log(posterior_leakage_samples)));
    % fprintf('Supremum of abs(log(Posterior Leakage)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum);
end


% Compute the average posterior leakage supremum over all locations
if isscalar(posterior_leakages_supremum)
    average_posterior_leakage = posterior_leakages_supremum;
else
    average_posterior_leakage = mean(posterior_leakages_supremum);
end
fprintf('Average Supremum of abs(log(Posterior Leakage)) over all locations: %.6f\n', average_posterior_leakage);

