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
    
    % Define target region bounds (from S/N of 20000 - 29999)
    TARGET_LON_MAX = 12.4528123; 
    TARGET_LON_MIN = 12.3727381; 
    TARGET_LAT_MAX = 41.8509768; 
    TARGET_LAT_MIN = 41.7857915;

    % Call function to find nodes within the target region
    [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN);
    
    % Generate obfuscated locations as the first 40% of the selected points
    num_obfuscated = max(1, floor(0.4 * length(original_longitude))); % Ensure at least one point is selected
    obfuscated_longitude = original_longitude(1:num_obfuscated);
    obfuscated_latitude = original_latitude(1:num_obfuscated);
    target_lat = col_latitude_selected(1);        % Target latitude
    target_long = col_longitude_selected(1);      % Target longitude

    % Define the target location (first element of the selected region data)
    target_location = [target_long, target_lat];  % Target location coordinates
    
    % Debug: Print the number of original and obfuscated nodes
    fprintf("Number of selected nodes: %d\n", length(col_osmid_selected));
    disp('Col_Osmid_Selected:');
    disp(col_osmid_selected);

    disp('Original Locations:');
    disp([original_longitude, original_latitude]);

    fprintf("Number of obfuscated nodes: %d\n", length(obfuscated_longitude));
    disp('Obfuscated Locations:');
    disp([obfuscated_longitude, obfuscated_latitude]);

    disp('Target Location:');
    disp(target_location);
    
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

    % Compute pairwise shortest path distances between original and target locations
    raw_distance_matrix_original = compute_raw_distance_matrix(original_longitude, original_latitude, target_long, target_lat, df_edges, df_nodes);

    % Compute pairwise shortest path distances between obfuscated and target locations
    raw_distance_matrix_obfuscated = compute_raw_distance_matrix(obfuscated_longitude, obfuscated_latitude, target_long, target_lat, df_edges, df_nodes);

    disp('Raw Distance Matrix (Original vs Target):');
    disp(raw_distance_matrix_original);

    disp('Raw Distance Matrix (Obfuscated vs Target):');
    disp(raw_distance_matrix_obfuscated);

    % Compute Cost Coefficient Matrix
    cost_coefficient = abs(raw_distance_matrix_original - raw_distance_matrix_obfuscated');

    disp('Cost Coefficient Matrix (Modulus of Differences):');
    disp(cost_coefficient);

    % Add noise to the Cost Coefficient Matrix
    epsilon = 1.0;
    sensitivity = 1.0;
    noisy_distance_matrix = add_noise_to_distance_matrix(cost_coefficient, sensitivity, epsilon);

    disp('Noisy Distance Matrix:');
    disp(noisy_distance_matrix);

    % Compute perturbation probabilities
    disp('Perturbation Probabilities:');
    perturbation_probabilities = compute_perturbation_probabilities(cost_coefficient, noisy_distance_matrix);
    disp(perturbation_probabilities);

    % Compute posterior probabilities
    disp('Posterior Probabilities:');
    posterior_probabilities_main = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, cost_coefficient, noisy_distance_matrix);
    disp(posterior_probabilities_main);

    % Compute 100 samples of noisy distance matrix and their posterior probabilities
    num_samples = 100;
    posterior_leakages = zeros(num_samples, 1);
    for sample_idx = 1:num_samples
        noisy_distance_sample = add_noise_to_distance_matrix(cost_coefficient, sensitivity, epsilon);
        posterior_probabilities_sample = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, cost_coefficient, noisy_distance_sample);
        posterior_leakages(sample_idx) = compute_posterior_leakage_value(posterior_probabilities_sample);
    end

    % Compute supremum of abs(log(posterior leakage))
    sup_abs_log_leakage = max(abs(log(posterior_leakages)));

    % Display results
    fprintf('Supremum of abs(log(Posterior Leakage)) from 100 samples: %.6f\n', sup_abs_log_leakage);
end

% Function to find nodes within the target region
function [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN)
    idx = (col_longitude >= TARGET_LON_MIN) & (col_longitude <= TARGET_LON_MAX) & ...
          (col_latitude >= TARGET_LAT_MIN) & (col_latitude <= TARGET_LAT_MAX);
    
    % Extract corresponding values
    [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx);

    % Randomly select 30 values using rng(0)
    [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, 30);
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



function raw_distance_matrix = compute_raw_distance_matrix(longitude, latitude, target_long, target_lat, df_edges, df_nodes)
    num_locations = length(longitude);

    % Create the graph
    G = create_graph_from_edges(df_edges, df_nodes);
    
    % Calculate shortest path distances between locations and target
    raw_distance_matrix = zeros(num_locations, 1);
    for i = 1:num_locations
        start_node = find_nearest_node(longitude(i), latitude(i), df_nodes);
        target_node = find_nearest_node(target_long, target_lat, df_nodes);

        % Use MATLAB's built-in shortest path function
        [~, raw_distance_matrix(i)] = shortestpath(G, start_node, target_node);  
    end
end


% Function to create a graph from the edges data
function G = create_graph_from_edges(df_edges, df_nodes)
    % Assuming 'df_edges' contains columns 'u', 'v', and 'length'
    
    id_list = df_nodes.osmid;
    
    edges_u = int64(df_edges.u);
    edges_v = int64(df_edges.v);
    edges_weight = df_edges.length;

    % Create graph object with weighted edges
    edges_u_index = zeros(size(edges_u));
    edges_v_index = zeros(size(edges_v));
    
    for i = 1:numel(edges_u)
        edges_u_index(i) = find(id_list == edges_u(i));
        edges_v_index(i) = find(id_list == edges_v(i));
    end
    
    G = graph(edges_u_index, edges_v_index, edges_weight);
end

% Function to find the nearest node based on Euclidean distance
function nearest_node = find_nearest_node(longitude, latitude, df_nodes)
    distances = sqrt((df_nodes.x - longitude).^2 + (df_nodes.y - latitude).^2);
    [~, nearest_node] = min(distances);
end


function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, sensitivity, epsilon)
    % Step 1: Compute the scale for Laplace noise
    scale = 0.1;

    % Step 2: Generate Laplace noise with the same size as distance_matrix
    noise_matrix = laplace_noise(scale, size(distance_matrix));

    % Step 3: Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
    noise_matrix = min(noise_matrix, distance_matrix - 0.1);

    % Step 4: Subtract the noise from the original distance matrix
    noisy_distance_matrix = distance_matrix - noise_matrix;
end

function noise = laplace_noise(scale, sz)
    uniform_random = rand(sz) - 0.5;
    noise = -scale * log(1 - 2 * abs(uniform_random));
end

function perturbation_probabilities = compute_perturbation_probabilities(raw_distance_matrix, noisy_distance_matrix)
    % Compute perturbation probabilities based on noisy and raw distances
    diff_matrix = abs(noisy_distance_matrix - raw_distance_matrix);
    b = 0.5; % Laplace noise scale
    perturbation_probabilities = (1 / (2 * b)) * exp(-diff_matrix / b);
end

function posterior_probabilities = compute_posterior_probabilities(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude, raw_distance_matrix, noisy_distance_matrix)
    num_original = length(original_longitude);
    num_obfuscated = length(obfuscated_longitude);

    % Define prior probabilities P(A) (uniform)
    P_A = ones(num_original, 1) / num_original;

    cardinality_N = num_original * num_obfuscated;

    % Define the noise model (Laplace noise)
    b = 0.5;

    % Compute likelihood P(B|A) for each pair of original and obfuscated locations
    P_B_given_A = zeros(num_original, num_obfuscated);
    for i = 1:num_original
        for j = 1:num_obfuscated
            dist = abs(noisy_distance_matrix(i, j) - raw_distance_matrix(i, j));
            P_B_given_A(i, j) = (1 / (2 * b))^cardinality_N * exp(-dist / b);
        end
    end

    % Compute the evidence P(B)
    P_B = sum(P_B_given_A .* P_A, 1); % Marginalize over all original locations

    % Compute posterior probabilities P(A|B)
    posterior_probabilities = (P_B_given_A .* P_A) ./ P_B;
end

function leakage_value = compute_posterior_leakage_value(posterior_probabilities)
    % Compute Posterior Leakage (PL)
    num_original = size(posterior_probabilities, 1);
    num_obfuscated = size(posterior_probabilities, 2);

    PL = zeros(num_original, num_obfuscated);

    % Iterate over all pairs of records (x_n, x_m)
    for n = 1:num_original
        for m = 1:num_obfuscated
            % Compute posterior ratio and prior ratio
            posterior_ratio = posterior_probabilities(n, :) ./ posterior_probabilities(m, :);
            prior_ratio = 1; % Uniform prior

            log_term = log(posterior_ratio / prior_ratio);

            % Take supremum (max value) over all y (columns in posterior probabilities)
            PL(n, m) = max(abs(log_term));
        end
    end

    leakage_value = max(PL(:));
end
