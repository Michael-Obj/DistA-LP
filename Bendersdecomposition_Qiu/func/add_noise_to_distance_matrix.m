function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon, clip_k)

        scale = 1 / epsilon;  % Laplace scale
        noise = laplace_noise(scale, size(distance_matrix));  % zero-mean Laplace

    if clip_k == "params"
        clip_k = 5;
        if nargin >= 3 && ~isempty(clip_k)
            lo = -clip_k * scale; hi = clip_k * scale;
            noise = max(min(noise, hi), lo);  % symmetric clipping (no bias)
        end
        noisy_distance_matrix = distance_matrix + noise;  % ADD noise, don't subtract
    
    else       
        % Step 3: Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
        noise_matrix = min(noise, distance_matrix - 0.1);
    
        % Step 4: Subtract the noise from the original distance matrix
        noisy_distance_matrix = distance_matrix - noise_matrix;
    end  
end

