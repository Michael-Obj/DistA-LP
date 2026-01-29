% using real data, worked with both original locations and obfuscated locations

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
    TARGET_LON_MAX = 12.375; 
    TARGET_LON_MIN = 12.2394; 
    TARGET_LAT_MAX = 42.0979; 
    TARGET_LAT_MIN = 41.888;

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

    % Compute pairwise Haversine distances
    raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude);

    disp('Raw Distance Matrix (Original vs Obfuscated):');
    disp(raw_distance_matrix);

    % Add noise to the distance matrix
    epsilon = 1.0;
    sensitivity = 1.0;
    noisy_distance_matrix = add_noise_to_distance_matrix(raw_distance_matrix, sensitivity, epsilon);

    disp('Noisy Distance Matrix:');
    disp(noisy_distance_matrix);

    % Compute perturbation probabilities
    disp('Perturbation Probabilities:');
    perturbation_probabilities = compute_perturbation_probabilities(raw_distance_matrix, noisy_distance_matrix);
    disp(perturbation_probabilities);

    % Compute posterior probabilities
    disp('Posterior Probabilities:');
    posterior_probabilities_main = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, raw_distance_matrix, noisy_distance_matrix, epsilon);
    disp(posterior_probabilities_main);

    % % Compute 100 samples of noisy distance matrix and their posterior probabilities
    % num_samples = 100;
    % posterior_leakages = zeros(num_samples, 1);
    % for sample_idx = 1:num_samples
    %     noisy_distance_sample = add_noise_to_distance_matrix(raw_distance_matrix, sensitivity, epsilon);
    %     posterior_probabilities_sample = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, raw_distance_matrix, noisy_distance_sample);
    %     posterior_leakages(sample_idx) = compute_posterior_leakage_value(posterior_probabilities_sample);
    % end
    % 
    % % Compute supremum of abs(log(posterior leakage))
    % sup_abs_log_leakage = max(abs(log(posterior_leakages)));
    % 
    % fprintf('Supremum of abs(log(Posterior Leakage)) from 100 samples: %.6f\n', sup_abs_log_leakage);


    % Number of times to repeat the entire sampling process
    num_iterations = 10;
    sup_abs_log_leakage_values = zeros(num_iterations, 1);
    
    for iteration_idx = 1:num_iterations
        % Compute 100 samples of noisy distance matrix and their posterior probabilities
        num_samples = 10;
        posterior_leakages = zeros(num_samples, 1);
        
        for sample_idx = 1:num_samples
            epsilon = sample_idx; 
            noisy_distance_sample = add_noise_to_distance_matrix(raw_distance_matrix, sensitivity, epsilon);
            posterior_probabilities_sample = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, raw_distance_matrix, noisy_distance_sample, epsilon);
            
            posterior_probabilities(sample_idx) = mean(mean(posterior_probabilities_sample)); 
            posterior_leakages(sample_idx) = compute_posterior_leakage_value(posterior_probabilities_sample);
        end
    
        % Compute supremum of abs(log(posterior leakage))
        sup_abs_log_leakage_values(iteration_idx) = max(abs(log(posterior_leakages)));
        
        fprintf('Iteration %d: Supremum of abs(log(Posterior Leakage)) = %.6f\n', iteration_idx, sup_abs_log_leakage_values(iteration_idx));
    end
    
    % Compute the average supremum of abs(log posterior leakage)
    average_sup_abs_log_leakage = mean(sup_abs_log_leakage_values);
    
    fprintf('Average Supremum of abs(log(Posterior Leakage)) over %d iterations: %.6f\n', num_iterations, average_sup_abs_log_leakage);
end

% Function to find nodes within the target region
function [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN)
    idx = (col_longitude >= TARGET_LON_MIN) & (col_longitude <= TARGET_LON_MAX) & ...
          (col_latitude >= TARGET_LAT_MIN) & (col_latitude <= TARGET_LAT_MAX);

    fprintf('Number of locations in the selected region: %d\n', length(idx));
    
    % Extract corresponding values
    [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx);

    % Randomly select 10% of the selected locations using rng(0)
    num_to_select = max(1, floor(0.01 * length(col_osmid_selected))); 
    [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, num_to_select);

    % % Randomly select 30 values using rng(0)
    % [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, 200);
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

% function d = haversine_distance(lat1, lon1, lat2, lon2)
%     R = 6371; % Earth's radius in km
%     dlat = deg2rad(lat2 - lat1);
%     dlon = deg2rad(lon2 - lon1);
%     a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2;
%     c = 2 * atan2(sqrt(a), sqrt(1 - a));
%     d = R * c;
% end

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

function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, sensitivity, epsilon)
    % Step 1: Compute the scale for Laplace noise
    scale = 1/epsilon;

    % Step 2: Generate Laplace noise with the same size as distance_matrix
    noise_matrix = laplace_noise(scale, size(distance_matrix));

    % Step 3: Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
    noise_matrix = min(noise_matrix, distance_matrix - 0.1);
    mean(mean(noise_matrix))

    % Step 4: Subtract the noise from the original distance matrix
    noisy_distance_matrix = distance_matrix - noise_matrix;
end

function noise = laplace_noise(scale, sz)
    uniform_random = rand(sz) - 0.5;
    noise = -scale * log(1 - 2 * abs(uniform_random));
end

function perturbation_probabilities = compute_perturbation_probabilities(raw_distance_matrix, noisy_distance_matrix)
    diff_matrix = abs(noisy_distance_matrix - raw_distance_matrix);
    b = 0.5;
    perturbation_probabilities = (1 / (2 * b)) * exp(-diff_matrix / b);
end

function posterior_probabilities = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, raw_distance_matrix, noisy_distance_matrix, epsilon)
    num_original = length(original_longitude);
    num_obfuscated = length(obfuscated_longitude);
    P_A = ones(num_original, 1) / num_original;
    cardinality_N = num_original * num_obfuscated;
    % b = 0.5;
    b = 1/epsilon; 
    P_B_given_A = zeros(num_original, num_obfuscated);
    for i = 1:num_original
        for j = 1:num_obfuscated
            dist = abs(noisy_distance_matrix(i, j) - raw_distance_matrix(i, j));
            P_B_given_A(i, j) = (1 / (2 * b))^cardinality_N * exp(-dist / b);
        end
    end
    P_B = sum(P_B_given_A .* P_A, 1);
    posterior_probabilities = (P_B_given_A .* P_A) ./ P_B;
end

function leakage_value = compute_posterior_leakage_value(posterior_probabilities)
    num_original = size(posterior_probabilities, 1);
    num_obfuscated = size(posterior_probabilities, 2);
    PL = zeros(num_original, num_obfuscated);
    for n = 1:num_original
        for m = 1:num_obfuscated
            posterior_ratio = posterior_probabilities(n, :) ./ posterior_probabilities(m, :);
            prior_ratio = 1;   % Since they are the same

            log_term = log(posterior_ratio / prior_ratio); % Take log of the ratios
            PL(n, m) = max(abs(log_term));
        end
    end
    leakage_value = max(PL(:));
end
