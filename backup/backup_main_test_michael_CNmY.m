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
title('Selected, Original, Obfuscated & Target Nodes in Target Region');
grid on;
hold off;

% Compute pairwise Haversine distances and build raw and noisy distance matrices for each location
num_locations = length(original_longitude);  % Number of original locations
num_obf = length(obfuscated_longitude);  % Number of obfuscated locations
raw_distance_matrix_original = cell(num_locations, 1);  % Cell array to store the 10x10 matrices
raw_distance_matrix_obfuscated = cell(num_obf, 1); 
raw_distance_matrices = cell(num_locations, 1);  % Cell array to store the 10x10 matrices
noisy_distance_matrices = cell(num_locations, 1); % Cell array to store the noisy 10x10 matrices
perturbation_probabilities = cell(num_locations, 1); % Cell array to store perturbation probabilities
posterior_probabilities = cell(num_locations, 1); % Cell array to store posterior probabilities


distance_matrix = compute_distance_matrix(original_latitude, original_longitude); 
% raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude);


%% Use captial letters for the constants 
EPSILON = 2;  % EPSILON value for noise
B_VALUE = 1/ EPSILON;  % B_VALUE for perturbation probability calculation
CARDINALITY_N = 10;




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

    % % Compute pairwise shortest path distances between original and target locations before alignment
    % raw_distance_matrix_original_unaligned = compute_raw_distance_matrix_cost(original_longitude(i), original_latitude(i), target_long, target_lat, df_edges, df_nodes);

    % % Compute pairwise shortest path distances between obfuscated and target locations
    % raw_distance_matrix_obfuscated_unaligned = compute_raw_distance_matrix_cost(obfuscated_longitude, obfuscated_latitude, target_long, target_lat, df_edges, df_nodes);




    % Store the aligned distance matrix in the cell array
    raw_distance_matrix_original{i} = raw_distance_matrix_original_unaligned;
    

    % % Store the aligned distance matrix in the cell array
    % raw_distance_matrix_obfuscated = raw_distance_matrix_obfuscated_unaligned;
    % 
    % 
    % % Compute Cost Coefficient Matrix
    % cost_coefficient{i} = abs(raw_distance_matrix_original{i} - raw_distance_matrix_obfuscated);


    % Display Results
    fprintf('Results for Location %d:\n', i);
    fprintf('Raw Distance Matrix (Original vs Target):\n');
    disp(raw_distance_matrix_original{i});

    % fprintf('Raw Distance Matrix (Obfuscated vs Target):\n');
    % disp(raw_distance_matrix_obfuscated);
    % 
    % fprintf('Cost Coefficient Matrix (Modulus of Differences):\n');
    % disp(cost_coefficient{i});
end


% Compute pairwise shortest path distances between obfuscated and target locations
raw_distance_matrix_obfuscated_unaligned = compute_raw_distance_matrix_cost(obfuscated_longitude, obfuscated_latitude, target_long, target_lat, df_edges, df_nodes);


fprintf('Raw Distance Matrix (Obfuscated vs Target):\n');
disp(raw_distance_matrix_obfuscated_unaligned);



for idx = 1:num_locations
    % It is assumed to be a vector of length CARDINALITY_N.
    original_distance_vector = raw_distance_matrix_original{idx};

    cost_coefficient = zeros(CARDINALITY_N, num_obf);

    % Compute the cost coefficient matrix element-wise. Loop over each element in the original vector (neighbors) and each obfuscated distance.
    for i = 1:CARDINALITY_N
        for j = 1:num_obf
            cost_coefficient(i, j) = abs(original_distance_vector(i) - raw_distance_matrix_obfuscated_unaligned(j));
        end
    end

    raw_distance_matrices{idx} = cost_coefficient;
    
    % Display the results for the current location.
    fprintf('Results for Location %d:\n', idx);
    fprintf('Cost Coefficient Matrix (Modulus of Differences):\n');
    disp(raw_distance_matrices{idx});
end





%% Compute global min and max from all aligned matrices
globalMin = inf;
globalMax = -inf;
for i = 1:num_locations
    curMat = raw_distance_matrices{i};
    globalMin = min(globalMin, min(curMat(:)));
    globalMax = max(globalMax, max(curMat(:)));
end


% Preallocate vector for average leakage for each quantization configuration
average_posterior_leakage = zeros(12, 1);
pooled = [];


% Loop over desired number of intervals from 1 to 12
for numIntervals = 1:12
    % Compute equally spaced boundaries; there will be numIntervals+1 boundaries.
    boundaries = linspace(globalMin, globalMax, numIntervals+1);
    
    % Build the intervals matrix: each row is [lower_bound, upper_bound, representative_value]
    intervals = zeros(numIntervals, 3);
    for j = 1:numIntervals
        intervals(j,1) = boundaries(j);      % lower bound
        intervals(j,2) = boundaries(j+1);      % upper bound
        if boundaries(j) == 0
            intervals(j,3) = (boundaries(j) + boundaries(j+1)) / 2; % use average if lower bound is 0
        else
            intervals(j,3) = boundaries(j);   % otherwise use the lower bound
        end
    end



    aligned_raw_distance_matrices = cell(size(raw_distance_matrices));
    for i = 1:num_locations
        aligned_raw_distance_matrices{i} = quantize_matrix_values(raw_distance_matrices{i}, intervals);
    end
    

    [quantized_matrices, row_perms, col_perms] = align_matrices(aligned_raw_distance_matrices, ...
                                                           100, ...       % max_iterations
                                                           1e-10);        % tolerance
    
    
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
            pooled = [pooled; PL_matrix{sample_idx}(:)];
        end
    
        disp('Posterior Probability Sample:');
        disp(posterior_prob_sample{sample_idx_});
       
        % Store the posterior leakages for this location
        % posterior_leakages(i, :) = posterior_leakage_samples';
    
        % % Print all posterior leakages for this location
        % fprintf('Posterior Leakages for Location %d:\n', i);
        % disp(posterior_leakage_samples);
        
        % Compute the supremum (max) of the absolute log of posterior leakages for this location
        posterior_leakages_supremum(i) = max(PL_max(i, :));
        fprintf('Supremum of abs(log(Posterior Leakage)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum(i));
    end
    
    
    % % Compute the average posterior leakage supremum over all locations
    % if isscalar(posterior_leakages_supremum)
    %     average_posterior_leakage = posterior_leakages_supremum;
    % else
    average_posterior_leakage(numIntervals) = mean(posterior_leakages_supremum);
    % end
    fprintf('For %d intervals, the Average Supremum of abs(log(Posterior Leakage)) over all locations is: %.6f\n', numIntervals, average_posterior_leakage(numIntervals));
end


disp("All Average Posterior Leakages with each Interval from 1 to 12")
disp(average_posterior_leakage)



% Plot the relationship between the number of intervals and the average posterior leakage.
figure;
plot(1:12, average_posterior_leakage, 's-', 'LineWidth', 2);
xlabel('Number of Quantization Intervals');
ylabel('Average Supremum of abs(log(Posterior Leakage))');
title('Posterior Leakage vs. Number of Intervals');
grid on;




figure; [f, xi] = ksdensity(pooled);
plot(xi, f, 'LineWidth', 2);
xlabel('abs(log(Posterior Leakage))');
ylabel('Density');
title('Aggregate Posterior Leakage (all entries)');
grid on;

