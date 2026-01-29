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

noisy_distance_matrices = cell(num_locations, 1); % Cell array to store the noisy 10x10 matrices
perturbation_probabilities = cell(num_locations, 1); % Cell array to store perturbation probabilities
posterior_probabilities = cell(num_locations, 1); % Cell array to store posterior probabilities



distance_matrix = compute_distance_matrix(original_latitude, original_longitude); 
% raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude);


%% Use captial letters for the constants 
EPSILON = 2;  % EPSILON value for noise
B_VALUE = 1/ EPSILON;  % B_VALUE for perturbation probability calculation
CARDINALITY_N = 10;
lambda2 = 1.0;
lambda3 = 1.0;









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



% Compute pairwise shortest path distances between obfuscated and target locations
cost_distance_matrix_obfuscated = compute_raw_distance_matrix_cost(obfuscated_longitude, obfuscated_latitude, target_long, target_lat, df_edges, df_nodes);

fprintf('Obfuscated Cost Distance Matrix (Obfuscated vs Target):\n');
disp(cost_distance_matrix_obfuscated);



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










%% Compute global min and max from all aligned matrices
globalMin = inf;
globalMax = -inf;
for i = 1:num_locations
    curMat_1 = original_distance_matrices{i};
    globalMin_1 = min(globalMin, min(curMat_1(:)));
    globalMax_1 = max(globalMax, max(curMat_1(:)));

    curMat_2 = obfuscated_distance_matrices{i};
    globalMin_2 = min(globalMin, min(curMat_2(:)));
    globalMax_2 = max(globalMax, max(curMat_2(:)));

    curMat_3 = cost_coefficient_matrices{i};
    globalMin_3 = min(globalMin, min(curMat_3(:)));
    globalMax_3 = max(globalMax, max(curMat_3(:)));
end

% Preallocate vector for average leakage for each quantization configuration
average_posterior_leakage_1 = zeros(12, 1);
pooled_1 = [];

average_posterior_leakage_2 = zeros(12, 1);
pooled_2 = [];

average_posterior_leakage_3 = zeros(12, 1);
pooled_3 = [];


% Loop over desired number of intervals from 1 to 12
for numIntervals = 1:12
    % Compute equally spaced boundaries; there will be numIntervals+1 boundaries.
    boundaries_1 = linspace(globalMin_1, globalMax_1, numIntervals+1);
    boundaries_2 = linspace(globalMin_2, globalMax_2, numIntervals+1);
    boundaries_3 = linspace(globalMin_3, globalMax_3, numIntervals+1);

    
    % Build the intervals matrix: each row is [lower_bound, upper_bound, representative_value]
    intervals_1 = zeros(numIntervals, 3);
    intervals_2 = zeros(numIntervals, 3);
    intervals_3 = zeros(numIntervals, 3);
    for j = 1:numIntervals
        intervals_1(j,1) = boundaries_1(j);      % lower bound
        intervals_1(j,2) = boundaries_1(j+1);      % upper bound
        if boundaries_1(j) == 0
            intervals_1(j,3) = (boundaries_1(j) + boundaries_1(j+1)) / 2; % use average if lower bound is 0
        else
            intervals_1(j,3) = boundaries_1(j);   % otherwise use the lower bound
        end


        intervals_2(j,1) = boundaries_2(j);      % lower bound
        intervals_2(j,2) = boundaries_2(j+1);      % upper bound
        if boundaries_2(j) == 0
            intervals_2(j,3) = (boundaries_2(j) + boundaries_2(j+1)) / 2; % use average if lower bound is 0
        else
            intervals_2(j,3) = boundaries_2(j);   % otherwise use the lower bound
        end


        intervals_3(j,1) = boundaries_3(j);      % lower bound
        intervals_3(j,2) = boundaries_3(j+1);      % upper bound
        if boundaries_3(j) == 0
            intervals_3(j,3) = (boundaries_3(j) + boundaries_3(j+1)) / 2; % use average if lower bound is 0
        else
            intervals_3(j,3) = boundaries_3(j);   % otherwise use the lower bound
        end
    end










    A1_all = cell(size(original_distance_matrices));
    for i = 1:num_locations
        A1_all{i} = quantize_matrix_values(original_distance_matrices{i}, intervals_1);
    end
    
    
    A2_all = cell(size(obfuscated_distance_matrices));
    for i = 1:num_locations
        A2_all{i} = quantize_matrix_values(obfuscated_distance_matrices{i}, intervals_2);
    end
    
    
    A3_all = cell(size(cost_coefficient_matrices));
    for i = 1:num_locations
        A3_all{i} = quantize_matrix_values(cost_coefficient_matrices{i}, intervals_3);
    end
    
    
    
    
    % Align user matrices
    [aligned_A1, aligned_A2, aligned_A3, best_pi] = ...
        align_user_matrices(A1_all, A2_all, A3_all, lambda2, lambda3);
    
    fprintf('Best shared permutation: [%s]\n', num2str(best_pi));
    
    % Visualize original vs aligned for first user
    figure('Name', 'User 1 Alignment Visualization', 'Position', [100, 100, 1200, 600]);
    
    subplot(2, 3, 1); imagesc(A1_all{1}); title('Original A1^{(1)}'); colorbar;
    subplot(2, 3, 2); imagesc(A2_all{1}); title('Original A2^{(1)}'); colorbar;
    subplot(2, 3, 3); imagesc(A3_all{1}); title('Original A3^{(1)}'); colorbar;
    
    subplot(2, 3, 4); imagesc(aligned_A1{1}); title('Aligned A1^{(1)}'); colorbar;
    subplot(2, 3, 5); imagesc(aligned_A2{1}); title('Aligned A2^{(1)}'); colorbar;
    subplot(2, 3, 6); imagesc(aligned_A3{1}); title('Aligned A3^{(1)}'); colorbar;
    
    







    
    % Number of locations (A, B, C, ..., T)
    % num_locations = 20;
    num_samples = 100; % Number of samples for each posterior probability
    posterior_leakages_1 = zeros(num_locations, num_samples);  % To store posterior leakages for each location
    posterior_leakages_2 = zeros(num_locations, num_samples);  % To store posterior leakages for each location
    posterior_leakages_3 = zeros(num_locations, num_samples);  % To store posterior leakages for each location    
    
    
    
    %% Calculate the posterior leakage
    % Iterate over all locations (A to T)
    for i = 1:num_locations    
        for sample_idx = 1:num_samples
            % Add noise to the distance matrix to the location i to generate a new noisy sample
            % EPSILON = sample_idx; 
            noisy_distance_sample = add_noise_to_distance_matrix(aligned_A1{i}, EPSILON);  % noiseType: 'laplace' or 'gaussian'
            
            % Compute the posterior probability for this noisy distance sample
            sample_idx_ = (i-1)*num_locations + sample_idx; 
            [posterior_prob_sample_1{sample_idx_}, PL_matrix{sample_idx_}, PL_max_1(i, sample_idx)] = compute_log_posterior(aligned_A1, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N);        
            pooled_1 = [pooled_1; PL_matrix{sample_idx}(:)];
        end


        for sample_idx = 1:num_samples
            % Add noise to the distance matrix to the location i to generate a new noisy sample
            % EPSILON = sample_idx; 
            noisy_distance_sample = add_noise_to_distance_matrix(aligned_A2{i}, EPSILON);  % noiseType: 'laplace' or 'gaussian'
            
            % Compute the posterior probability for this noisy distance sample
            sample_idx__ = (i-1)*num_locations + sample_idx; 
            [posterior_prob_sample_2{sample_idx__}, PL_matrix{sample_idx__}, PL_max_2(i, sample_idx)] = compute_log_posterior(aligned_A2, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N);        
            pooled_2 = [pooled_2; PL_matrix{sample_idx}(:)];
        end


        for sample_idx = 1:num_samples
            % Add noise to the distance matrix to the location i to generate a new noisy sample
            % EPSILON = sample_idx; 
            noisy_distance_sample = add_noise_to_distance_matrix(aligned_A3{i}, EPSILON);  % noiseType: 'laplace' or 'gaussian'
            
            % Compute the posterior probability for this noisy distance sample
            sample_idx___ = (i-1)*num_locations + sample_idx; 
            [posterior_prob_sample_3{sample_idx___}, PL_matrix{sample_idx___}, PL_max_3(i, sample_idx)] = compute_log_posterior(aligned_A3, noisy_distance_sample, distance_matrix, num_locations, B_VALUE, CARDINALITY_N);        
            pooled_3 = [pooled_3; PL_matrix{sample_idx}(:)];
        end


    
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
    average_posterior_leakage_1(numIntervals) = mean(posterior_leakages_supremum_1);
    % end
    fprintf('For %d intervals, the Average Supremum of abs(log(Posterior Leakage 1)) over all locations is: %.6f\n', numIntervals, average_posterior_leakage_1(numIntervals));



    average_posterior_leakage_2(numIntervals) = mean(posterior_leakages_supremum_2);
    fprintf('For %d intervals, the Average Supremum of abs(log(Posterior Leakage 2)) over all locations is: %.6f\n', numIntervals, average_posterior_leakage_2(numIntervals));


    average_posterior_leakage_3(numIntervals) = mean(posterior_leakages_supremum_3);
    fprintf('For %d intervals, the Average Supremum of abs(log(Posterior Leakage 3)) over all locations is: %.6f\n', numIntervals, average_posterior_leakage_3(numIntervals));
end

disp("All Average Posterior Leakages 1 with each Interval from 1 to 12")
disp(average_posterior_leakage_1)

disp("All Average Posterior Leakages 2 with each Interval from 1 to 12")
disp(average_posterior_leakage_2)

disp("All Average Posterior Leakages 3 with each Interval from 1 to 12")
disp(average_posterior_leakage_3)





% Plot the relationship between the number of intervals and the average posterior leakage.
figure;
plot(1:12, average_posterior_leakage_1, 's-', 'LineWidth', 2);
xlabel('Number of Quantization Intervals');
ylabel('Average Supremum of abs(log(Posterior Leakage))');
title('Posterior Leakage 1 vs. Number of Intervals');
grid on;


% Plot the relationship between the number of intervals and the average posterior leakage.
figure;
plot(1:12, average_posterior_leakage_2, 's-', 'LineWidth', 2);
xlabel('Number of Quantization Intervals');
ylabel('Average Supremum of abs(log(Posterior Leakage))');
title('Posterior Leakage 2 vs. Number of Intervals');
grid on;


% Plot the relationship between the number of intervals and the average posterior leakage.
figure;
plot(1:12, average_posterior_leakage_3, 's-', 'LineWidth', 2);
xlabel('Number of Quantization Intervals');
ylabel('Average Supremum of abs(log(Posterior Leakage))');
title('Posterior Leakage 3 vs. Number of Intervals');
grid on;



% % PDF plot
% figure; [f, xi] = ksdensity(pooled);
% plot(xi, f, 'LineWidth', 2);
% xlabel('abs(log(Posterior Leakage))');
% ylabel('Density');
% title('Aggregate Posterior Leakage (all entries)');
% grid on;



% % CDF plot
% figure;
% [f_cdf, xi_cdf] = ksdensity(pooled, 'Function', 'cdf');
% plot(xi_cdf, f_cdf, 'LineWidth', 2);
% xlabel('abs(log(Posterior Leakage))');
% ylabel('Cumulative Probability');
% title('CDF of Posterior Leakage');
% grid on;





% Suppose 'average_posterior_leakage' is your 12×1 vector:
avg_val = mean(average_posterior_leakage_1);

% Build the CDF of whatever pooled data you have:
figure;
[f_cdf, xi_cdf] = ksdensity(pooled_1, 'Function', 'cdf');
plot(xi_cdf, f_cdf, 'LineWidth', 2);
hold on;

% Find the CDF value at the mean via interpolation:
cdf_at_mean = interp1(xi_cdf, f_cdf, avg_val);

% Draw a vertical dashed line at x = avg_val from y = 0 up to the CDF:
plot([avg_val, avg_val], [0, cdf_at_mean], '--r', 'LineWidth', 1.5);

% (Optional) Draw a horizontal dashed line at the CDF level back to x-axis:
plot([min(xi_cdf), avg_val], [cdf_at_mean, cdf_at_mean], '--r', 'LineWidth', 1.5);

% Mark the intersection point and label it:
scatter(avg_val, cdf_at_mean, 50, 'r', 'filled');
text(avg_val, cdf_at_mean, sprintf('  Mean = %.4f', avg_val), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

hold off;
xlabel('abs(log(Posterior Leakage 1))');
ylabel('Cumulative Probability 1');
title('CDF of Posterior Leakage 1 with Overall Mean');
grid on;
% ===================================================================================



% Suppose 'average_posterior_leakage' is your 12×1 vector:
avg_val_2 = mean(average_posterior_leakage_2);

% Build the CDF of whatever pooled data you have:
figure;
[f_cdf_2, xi_cdf_2] = ksdensity(pooled_2, 'Function', 'cdf');
plot(xi_cdf_2, f_cdf_2, 'LineWidth', 2);
hold on;

% Find the CDF value at the mean via interpolation:
cdf_at_mean_2 = interp1(xi_cdf_2, f_cdf_2, avg_val_2);

% Draw a vertical dashed line at x = avg_val from y = 0 up to the CDF:
plot([avg_val_2, avg_val_2], [0, cdf_at_mean_2], '--r', 'LineWidth', 1.5);

% (Optional) Draw a horizontal dashed line at the CDF level back to x-axis:
plot([min(xi_cdf_2), avg_val_2], [cdf_at_mean_2, cdf_at_mean_2], '--r', 'LineWidth', 1.5);

% Mark the intersection point and label it:
scatter(avg_val_2, cdf_at_mean_2, 50, 'r', 'filled');
text(avg_val_2, cdf_at_mean_2, sprintf('  Mean = %.4f', avg_val_2), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

hold off;
xlabel('abs(log(Posterior Leakage 2))');
ylabel('Cumulative Probability 2');
title('CDF of Posterior Leakage 2 with Overall Mean');
grid on;
% ===================================================================================



% Suppose 'average_posterior_leakage' is your 12×1 vector:
avg_val_3 = mean(average_posterior_leakage_3);

% Build the CDF of whatever pooled data you have:
figure;
[f_cdf_3, xi_cdf_3] = ksdensity(pooled_3, 'Function', 'cdf');
plot(xi_cdf_3, f_cdf_3, 'LineWidth', 2);
hold on;

% Find the CDF value at the mean via interpolation:
cdf_at_mean_3 = interp1(xi_cdf_3, f_cdf_3, avg_val_3);

% Draw a vertical dashed line at x = avg_val from y = 0 up to the CDF:
plot([avg_val_3, avg_val_3], [0, cdf_at_mean_3], '--r', 'LineWidth', 1.5);

% (Optional) Draw a horizontal dashed line at the CDF level back to x-axis:
plot([min(xi_cdf_3), avg_val_3], [cdf_at_mean_3, cdf_at_mean_3], '--r', 'LineWidth', 1.5);

% Mark the intersection point and label it:
scatter(avg_val_3, cdf_at_mean_3, 50, 'r', 'filled');
text(avg_val_3, cdf_at_mean_3, sprintf('  Mean = %.4f', avg_val_3), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

hold off;
xlabel('abs(log(Posterior Leakage 3))');
ylabel('Cumulative Probability 3');
title('CDF of Posterior Leakage 3 with Overall Mean');
grid on;
