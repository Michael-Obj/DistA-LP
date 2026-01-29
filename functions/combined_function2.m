function combined_function()
    % Generate random locations
    num_points = 10;
    columns = 3;
    locations = rand(num_points, columns);

    % Compute the distance matrix
    [raw_distance_matrix, symmetric_normalized_matrix] = compute_symmetric_normalized_matrix(locations, num_points);

    disp('Raw Distance Matrix:');
    disp(raw_distance_matrix);
    
    disp('Symmetric Normalized Distance Matrix:');
    disp(symmetric_normalized_matrix);

    x = locations(1, :);
    y = locations(2, :);
 
    epsilon = 1.0;
    sensitivity = compute_local_sensitivity(x, y, locations);

    % Laplace noise with threshold
    noisy_distance = apply_laplace_mechanism_with_threshold(x, y, sensitivity, epsilon);

    % Results
    original_distance = euclidean_distance(x, y);
    fprintf('Original Distance: %.6f\n', original_distance);
    fprintf('Noisy Distance: %.6f\n', noisy_distance);
end

function [raw_distance_matrix, symmetric_normalized_matrix] = compute_symmetric_normalized_matrix(locations, num_points)
    % Initialize raw distance matrix
    raw_distance_matrix = zeros(num_points, num_points);

    % Calculate pairwise Euclidean distances
    for i = 1:num_points
        for j = 1:num_points
            raw_distance_matrix(i, j) = distance(locations(i, :), locations(j, :));
        end
    end

    % Compute symmetric normalized matrix
    symmetric_normalized_matrix = raw_distance_matrix;
    total_sum = sum(raw_distance_matrix(:));
    if total_sum > 0
        symmetric_normalized_matrix = raw_distance_matrix / total_sum;
    end
end

function d = distance(point1, point2)
    % Compute Euclidean distance between two points
    d = sqrt(sum((point1 - point2).^2));
end

function d = euclidean_distance(x, y)
    % Distance between two points
    d = sqrt(sum((x - y).^2));
end

function sensitivity = compute_local_sensitivity(x, y, locations)
    % Local sensitivity for the given pair (x, y)
    d_x_y = euclidean_distance(x, y);
    sensitivities = zeros(size(locations, 1), 1);

    for i = 1:size(locations, 1)
        z = locations(i, :);
        sensitivities(i) = abs(d_x_y - euclidean_distance(y, z));
    end

    sensitivity = max(sensitivities);
end

function noisy_distance = apply_laplace_mechanism_with_threshold(x, y, sensitivity, epsilon)
    % Laplace mechanism with a threshold to ensure noisy_distance >= original_distance
    original_distance = euclidean_distance(x, y);
    scale = sensitivity / epsilon;

    % Initialize noisy_distance
    noisy_distance = original_distance;

    % Recompute noise until the noisy distance is valid
    while noisy_distance >= original_distance || noisy_distance <= 0
        noise = laplace_noise(scale);
        noisy_distance = original_distance - noise;
    end
end

function noise = laplace_noise(scale)
    % Laplace noise with given scale
    uniform_random = rand - 0.5; % Uniform random value between -0.5 and 0.5
    noise = -scale * log(1 - 2 * abs(uniform_random)); % Generate Laplace noise
end
