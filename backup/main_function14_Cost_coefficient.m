% working with both original locations, obfuscated and target locations
% calculated the cost coefficient
% Poesterior Leakage with 100 samples of the noisy distance matrix

function main_function()
    num_original_points = 10;
    num_obfuscated_points = 5;
    columns = 3;

    % Generate random original, obfuscated, and target locations
    original_locations = rand(num_original_points, columns);
    obfuscated_locations = rand(num_obfuscated_points, columns);
    target_location = rand(1, columns);

    disp('Original Locations:');
    disp(original_locations);
    disp('Obfuscated Locations:');
    disp(obfuscated_locations);
    disp('Target Location:');
    disp(target_location);

    % Compute pairwise distances between original and target locations
    raw_distance_matrix_original = compute_raw_distance_matrix(original_locations, target_location);

    % Compute pairwise distances between obfuscated and target locations
    raw_distance_matrix_obfuscated = compute_raw_distance_matrix(obfuscated_locations, target_location);

    

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
    posterior_probabilities_main = compute_posterior_probabilities(original_locations, obfuscated_locations, cost_coefficient, noisy_distance_matrix);
    disp(posterior_probabilities_main);

    % Compute 100 samples of noisy distance matrix and their posterior probabilities
    num_samples = 100;
    posterior_leakages = zeros(num_samples, 1);
    for sample_idx = 1:num_samples
        noisy_distance_sample = add_noise_to_distance_matrix(cost_coefficient, sensitivity, epsilon);
        posterior_probabilities_sample = compute_posterior_probabilities(original_locations, obfuscated_locations, cost_coefficient, noisy_distance_sample);
        posterior_leakages(sample_idx) = compute_posterior_leakage_value(posterior_probabilities_sample);
    end

    % Compute supremum of abs(log(posterior leakage))
    sup_abs_log_leakage = max(abs(log(posterior_leakages)));

    % Display results
    fprintf('Supremum of abs(log(Posterior Leakage)) from 100 samples: %.6f\n', sup_abs_log_leakage);
end

function raw_distance_matrix = compute_raw_distance_matrix(locations, target_location)
    num_locations = size(locations, 1);

    % Calculate pairwise Euclidean distances between locations and target
    raw_distance_matrix = zeros(num_locations, 1);
    for i = 1:num_locations
        raw_distance_matrix(i) = distance(locations(i, :), target_location);
    end
end

function noisy_distance_matrix = add_noise_to_distance_matrix(cost_coefficient, sensitivity, epsilon)
    noisy_distance_matrix = zeros(size(cost_coefficient));
    scale = sensitivity / epsilon; % Fixed Laplace noise scale

    % Add Laplace noise to each distance
    for i = 1:size(cost_coefficient, 1)
        for j = 1:size(cost_coefficient, 2)
            noise = laplace_noise(scale);
            % Ensure noise is positive and capped at original distance
            noise = min(noise, cost_coefficient(i, j) - 0.1);
            noisy_distance_matrix(i, j) = cost_coefficient(i, j) - noise;
        end
    end
end

function noise = laplace_noise(scale)
    % Generate Laplace noise
    uniform_random = rand - 0.5; % Uniform random value between -0.5 and 0.5
    noise = -scale * log(1 - 2 * abs(uniform_random));
end

function d = distance(point1, point2)
    d = sqrt(sum((point1 - point2).^2));
end

function perturbation_probabilities = compute_perturbation_probabilities(cost_coefficient, noisy_distance_matrix)
    % Compute perturbation probabilities based on noisy and raw distances
    diff_matrix = abs(noisy_distance_matrix - cost_coefficient);
    b = 0.5; % Laplace noise scale
    perturbation_probabilities = (1 / (2 * b)) * exp(-diff_matrix / b);
end

function posterior_probabilities = compute_posterior_probabilities(original_locations, obfuscated_locations, cost_coefficient, noisy_distance_matrix)
    num_original = size(original_locations, 1);
    num_obfuscated = size(obfuscated_locations, 1);

    % Define prior probabilities P(A) (uniform)
    P_A = ones(num_original, 1) / num_original;

    cardinality_N = num_original * num_obfuscated;

    % Define the noise model (Laplace noise)
    b = 0.5;

    % Compute likelihood P(B|A) for each pair of original and obfuscated locations
    P_B_given_A = zeros(num_original, num_obfuscated);
    for i = 1:num_original
        for j = 1:num_obfuscated
            dist = abs(noisy_distance_matrix(i, j) - cost_coefficient(i, j));
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