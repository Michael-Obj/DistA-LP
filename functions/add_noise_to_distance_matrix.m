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

