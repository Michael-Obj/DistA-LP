function main_function()
    % Load the dataset
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
    
    epsilon = 1.0;  % Epsilon value for noise
    b_value = 0.5;  % b value for perturbation probability calculation
    
    % Loop through all locations (A, B, C, ..., T)
    for i = 1:num_locations
        % Select the 9 nearest neighbors for the current location
        [nearest_longitude, nearest_latitude] = select_nearest_neighbors(original_longitude, original_latitude, i, 9);
        
        % Compute the raw distance matrix for this location and its 9 nearest neighbors
        raw_distance_matrix = compute_raw_distance_matrix([original_longitude(i); nearest_longitude], [original_latitude(i); nearest_latitude], ...
                                                           [obfuscated_longitude; nearest_longitude], [obfuscated_latitude; nearest_latitude]);
        
        % Store the distance matrix in the cell array
        raw_distance_matrices{i} = raw_distance_matrix;

        % Compute the noisy distance matrix for location i (D_i)
        noisy_distance_matrix = add_noise_to_distance_matrix(raw_distance_matrices{i}, epsilon);
        
        % Compute the perturbation probability P(D_i|i)
        perturbation_prob = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_matrix, b_value, 10);
        % Store perturbation probability for location i
        perturbation_probabilities{i} = perturbation_prob;
        
        % Compute the posterior probability P(i|D_i)
        posterior_prob = compute_posterior_probabilities(i, perturbation_prob, num_locations);
        % Store posterior probability for location i
        posterior_probabilities{i} = posterior_prob;
 
        % Display Results
        fprintf('Results for Location %d:\n', i);
        fprintf('Raw Distance Matrix:\n');
        disp(raw_distance_matrices{i});
        fprintf('Noisy Distance Matrix:\n');
        disp(noisy_distance_matrix);
        fprintf('Perturbation Probability Matrix:\n');
        disp(perturbation_probabilities{i});
        fprintf('Posterior Probability Matrix:\n');
        disp(posterior_probabilities{i});
        fprintf('---------------\n');
    end


    % Number of locations (A, B, C, ..., T)
    num_locations = 20;
    num_samples = 100; % Number of samples for each posterior probability
    posterior_leakages = zeros(num_locations, num_samples);  % To store posterior leakages for each location
    
    % Iterate over all locations (A to T)
    for i = 1:num_locations
        % Compute the noisy distance matrix for the current location (D_i)
        noisy_distance_matrix = add_noise_to_distance_matrix(raw_distance_matrices{i}, epsilon);
        
        % Compute the perturbation probability P(D_i|i)
        perturbation_prob = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_matrix, b_value, 10);
        
        % Compute the posterior probability P(i|D_i) for this location
        posterior_probabilities = compute_posterior_probabilities(i, perturbation_prob, num_locations);
        
        % Sample the noisy distance matrix 100 times for posterior leakage computation
        posterior_leakage_samples = zeros(num_samples, 1);
        
        for sample_idx = 1:num_samples
            % Add noise to the distance matrix to generate a new noisy sample
            noisy_distance_sample = add_noise_to_distance_matrix(raw_distance_matrices{i}, epsilon);
            
            % Compute perturbation probability for this noisy distance sample
            perturbation_prob_sample = compute_perturbation_probabilities(raw_distance_matrices{i}, noisy_distance_sample, b_value, 10);
            
            % Compute the posterior probability for this noisy distance sample
            posterior_prob_sample = compute_posterior_probabilities(i, perturbation_prob_sample, num_locations);
            
            % Compute the posterior leakage for this sample
            posterior_leakage_samples(sample_idx) = compute_posterior_leakage_value(posterior_prob_sample, posterior_probabilities);
        end
        
        % Store the posterior leakages for this location
        posterior_leakages(i, :) = posterior_leakage_samples';
    
        % % Print all posterior leakages for this location
        % fprintf('Posterior Leakages for Location %d:\n', i);
        % disp(posterior_leakage_samples);
        
        % Compute the supremum (max) of the absolute log of posterior leakages for this location
        posterior_leakages_supremum = max(abs(log(posterior_leakage_samples)));
        fprintf('Supremum of abs(log(Posterior Leakage)) for Location %d = %.6f\n\n', i, posterior_leakages_supremum);
    end
    
    % Compute the average posterior leakage supremum over all locations
    if isscalar(posterior_leakages_supremum)
        average_posterior_leakage = posterior_leakages_supremum;
    else
        average_posterior_leakage = mean(posterior_leakages_supremum);
    end
    fprintf('Average Supremum of abs(log(Posterior Leakage)) over all locations: %.6f\n', average_posterior_leakage);
end




% Function to find nodes within the target region
function [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN)
    idx = (col_longitude >= TARGET_LON_MIN) & (col_longitude <= TARGET_LON_MAX) & ...
          (col_latitude >= TARGET_LAT_MIN) & (col_latitude <= TARGET_LAT_MAX);

    fprintf('Number of locations in the selected region: %d\n', length(idx));
    
    % Extract corresponding values
    [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx);

    % Randomly select 10% of the selected locations using rng(0) / Randomly select 20 values using rng(0)
    num_to_select = max(1, floor(0.02 * length(col_osmid_selected))); 
    [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, 20);
end




% Function to extract corresponding values
function [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx)
    col_osmid_selected = col_osmid(idx);
    col_longitude_selected = col_longitude(idx);
    col_latitude_selected = col_latitude(idx);
end




% Function to randomly select N nodes using rng(0)
function [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, n)
    rng(0);  % Set the random seed for reproducibility
    num_selected = min(n, length(col_osmid_selected));  % Ensure we don't select more than available data

    % Randomly select indices
    random_indices = randperm(length(col_osmid_selected), num_selected);
    
    % Select the random nodes
    col_osmid_selected = col_osmid_selected(random_indices);
    original_longitude = col_longitude_selected(random_indices);
    original_latitude = col_latitude_selected(random_indices);
end




% Function to select nearest neighbors for a given location
function [nearest_longitude, nearest_latitude] = select_nearest_neighbors(longitudes, latitudes, idx, num_neighbors)
    distances = zeros(length(longitudes), 1);
    for i = 1:length(longitudes)
        if i ~= idx
            distances(i) = haversine_distance(latitudes(idx), longitudes(idx), latitudes(i), longitudes(i));
        else
            distances(i) = inf; % Exclude the location itself
        end
    end
    
    [~, sorted_indices] = sort(distances);
    nearest_longitude = longitudes(sorted_indices(1:num_neighbors));
    nearest_latitude = latitudes(sorted_indices(1:num_neighbors));
end




% Function to compute raw distance matrix
function raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude)
    num_original = length(original_longitude);
    num_obfuscated = length(obfuscated_longitude);
    raw_distance_matrix = zeros(num_original, num_obfuscated);
    for i = 1:num_original
        for j = 1:num_obfuscated
            raw_distance_matrix(i, j) = haversine_distance(original_latitude(i), original_longitude(i), obfuscated_latitude(j), obfuscated_longitude(j));
        end
    end
end




% Haversine distance function
function d = haversine_distance(lat1, lon1, lat2, lon2)
    R = 6372.8; % Earth's radius in kilometers
    dLat = deg2rad(lat2 - lat1);
    dLon = deg2rad(lon2 - lon1);
    lat1 = deg2rad(lat1);
    lat2 = deg2rad(lat2);
    a = sin(dLat/2)^2 + sin(dLon/2)^2*cos(lat1)*cos(lat2);
    c = 2*asin(sqrt(a));
    d = R*c;
end




% Function to add Laplace noise to the distance matrix
function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon)
    % Step 1: Compute the scale for Laplace noise
    scale = 1 / epsilon;

    % Step 2: Generate Laplace noise with the same size as distance_matrix
    noise_matrix = laplace_noise(scale, size(distance_matrix));

    % Step 3: Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
    noise_matrix = min(noise_matrix, distance_matrix - 0.1);

    % Step 4: Subtract the noise from the original distance matrix
    noisy_distance_matrix = distance_matrix - noise_matrix;
end




% Function to generate Laplace noise
function noise = laplace_noise(scale, sz)
    uniform_random = rand(sz) - 0.5;
    noise = -scale * log(1 - 2 * abs(uniform_random));
end




% Function to compute perturbation probabilities
function perturbation_probabilities = compute_perturbation_probabilities(raw_distance_matrix, noisy_distance_matrix, b_value, cardinality_N)
    % Initialize the perturbation probabilities matrix
    perturbation_probabilities = zeros(size(raw_distance_matrix));
    
    % Calculate the perturbation probabilities for each element in the matrix
    for i = 1:size(raw_distance_matrix, 1)
        for j = 1:size(raw_distance_matrix, 2)
            diff = abs(noisy_distance_matrix(i, j) - raw_distance_matrix(i, j));
            perturbation_probabilities(i, j) = (1 / (2 * b_value))^cardinality_N * exp(-diff / b_value);
        end
    end
end




% Function to compute the posterior probability for each location
function posterior_prob = compute_posterior_probabilities(location_idx, perturbation_prob, num_locations)
    % Compute the sum of all perturbation probabilities weighted by priors
    prior_probability = 1 / num_locations;
    conditional_probability = zeros(1, num_locations);

    for i = 1:num_locations
        % P(D_i | i) * P(i) for each location
        conditional_probability(1, i) = conditional_probability(1, i) + perturbation_prob(i) * prior_probability;
    end
    
    % Calculate the posterior probability for the current location
    % posterior_prob = (perturbation_prob(location_idx) * prior_probability) / sum(conditional_probability(1, i));

    % Compute the sum of all conditional probabilities for normalization
    normalization_factor = sum(conditional_probability);
    
    % Calculate the posterior probabilities for all locations
    posterior_prob = conditional_probability / normalization_factor;
end




% Function to compute posterior leakage value for all locations
function leakage_value = compute_posterior_leakage_value(posterior_prob_sample, posterior_probabilities)
    % Initialize the posterior leakage matrix
    PL = zeros(size(posterior_prob_sample));

    % Loop through all rows (original locations) and columns (obfuscated locations)
    num_original = size(posterior_prob_sample, 1);  % Now this should be 20 (number of original locations)
    num_obfuscated = size(posterior_prob_sample, 2);  % Number of obfuscated locations (8, in this case)
    
    for n = 1:num_original
        for m = 1:num_obfuscated
            % Ensure that the indices are within bounds
            if n <= size(posterior_prob_sample, 1) && m <= size(posterior_probabilities, 2)
                % Compute posterior ratio and log term
                posterior_ratio = posterior_prob_sample(n, :) ./ posterior_probabilities(n, :); % Compare the same row (original location)
                log_term = log(abs(posterior_ratio)); % Take log of the ratio
                PL(n, m) = max(abs(log_term));  % Store the maximum log difference (posterior leakage)
            % else
                % In case of out-of-bounds indices (shouldn't happen)
                % fprintf('Index out of bounds: n = %d, m = %d\n', n, m);
            end
        end
    end

    % Return the maximum posterior leakage value
    leakage_value = max(PL(:));  % Max of all leakage values
end
