% Perturbation probability is added and P(B/A) is recomputed.

function main_function() 
    num_points = 10;
    columns = 3;
    locations = rand(num_points, columns); 

    disp('Original Locations:');
    disp(locations);

    % Compute the raw distance matrix
    raw_distance_matrix = compute_raw_distance_matrix(locations, num_points);

    disp('Raw Distance Matrix:');
    disp(raw_distance_matrix);

    % Add noise to the raw distance matrix
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
    fprintf('Posterior probabilities:\n');
    posterior_probabilities = compute_posterior_probabilities(locations, raw_distance_matrix, noisy_distance_matrix);

    % Compute Posterior Leakage (PL)
    fprintf('Posterior Leakage:\n');
    compute_posterior_leakage(posterior_probabilities);
end

function raw_distance_matrix = compute_raw_distance_matrix(locations, num_points)
    raw_distance_matrix = zeros(num_points, num_points);

    % Calculate pairwise Euclidean distances
    for i = 1:num_points
        for j = 1:num_points
            raw_distance_matrix(i, j) = distance(locations(i, :), locations(j, :));
        end
    end
end

function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, sensitivity, epsilon)
    noisy_distance_matrix = zeros(size(distance_matrix));

    scale = sensitivity / epsilon; % Fixed Laplace noise scale

    % Add Laplace noise to each distance
    for i = 1:size(distance_matrix, 1)
        for j = 1:size(distance_matrix, 2)
            noise = laplace_noise(scale);
            % Ensure noise is positive and capped at original distance
            noise = min(noise, distance_matrix(i, j) - 0.1);
            noisy_distance_matrix(i, j) = distance_matrix(i, j) - noise;
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

function perturbation_probabilities = compute_perturbation_probabilities(raw_distance_matrix, noisy_distance_matrix)
    % Compute perturbation probabilities based on noisy and raw distances
    diff_matrix = abs(noisy_distance_matrix - raw_distance_matrix);
    b = 0.5; % Laplace noise scale
    perturbation_probabilities = (1 / (2 * b)) * exp(-diff_matrix / b);
end

function posterior_probabilities = compute_posterior_probabilities(original_locations, raw_distance_matrix, noisy_distance_matrix)
    num_original = size(original_locations, 1);

    % Define prior probabilities P(A) (uniform)
    P_A = ones(num_original, 1) / num_original;

    cardinality_N = num_original * (num_original - 1) / 2;

    % Define the noise model (Laplace noise)
    b = 0.5; 

    % Compute likelihood P(B|A) for each pair of original locations and distances
    P_B_given_A = zeros(num_original, num_original);
    for i = 1:num_original
        for j = 1:num_original
            dist = abs(noisy_distance_matrix(i, j) - raw_distance_matrix(i, j));
            P_B_given_A(i, j) = (1 / (2 * b))^cardinality_N * exp(-dist / b);
        end
    end

    % Compute the evidence P(B)
    P_B = sum(P_B_given_A .* P_A, 1); % Marginalize over all original locations

    % Compute posterior probabilities P(A|B)
    posterior_probabilities = (P_B_given_A .* P_A) ./ P_B;

    % Display results
    disp(posterior_probabilities);
end

function compute_posterior_leakage(posterior_probabilities)
    % Compute Posterior Leakage (PL) 
    num_records = size(posterior_probabilities, 1);
    PL = zeros(num_records, num_records);

    % Iterate over all pairs of records (x_n, x_m)
    for n = 1:num_records
        for m = 1:num_records
            if n ~= m
                % Compute posterior ratio and prior ratio 
                posterior_ratio = posterior_probabilities(n, :) ./ posterior_probabilities(m, :);
                prior_ratio = 1;   % Since they are the same

                log_term = log(posterior_ratio / prior_ratio);

                % Take supremum (max value) over all y (columns in posterior probabilities)
                PL(n, m) = max(abs(log_term));
            end
        end
    end

    disp(PL);
end
