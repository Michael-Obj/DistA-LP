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






% % Function to add Laplace or gaussian noise to the distance matrix
% function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon)   % noiseType: 'laplace' or 'gaussian'
%     switch lower(noiseType)
%         case 'laplace'
%             % Laplace noise mechanism
%             scale = 1 / epsilon;
% 
%             % Generate Laplace noise with the same size as distance_matrix
%             noise_matrix = laplace_noise(scale, size(distance_matrix));
% 
%             % Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
%             noise_matrix = min(noise_matrix, distance_matrix - 0.1);
% 
%             % Subtract the noise from the original distance matrix
%             noisy_distance_matrix = distance_matrix - noise_matrix;
% 
%         case 'gaussian'
%             % Differential privacy Gaussian mechanism
%             delta = 0.01;      % small constant; adjust as needed
%             sensitivity = 1;   % adjust if necessary for your application
%             sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
% 
%             % Generate Gaussian noise with mean 0 and standard deviation sigma
%             noise = sigma * randn(size(distance_matrix));
% 
%             % Add the noise to the original distance matrix
%             noisy_distance_matrix = distance_matrix + noise;
% 
%             % Optionally enforce non-negativity
%             noisy_distance_matrix = max(noisy_distance_matrix, 0.1);
% 
%         otherwise
%             error('Unsupported noise type. Choose "laplace" or "gaussian".');
%     end
% end




% % Function to add Laplace noise to the distance matrix
% function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon)
%      % Step 1: Compute the scale for Laplace noise
%      scale = 1 / epsilon;
% 
%      % Step 2: Generate Laplace noise with the same size as distance_matrix
%      noise_matrix = laplace_noise(scale, size(distance_matrix));
% 
%      % Step 3: Ensure the noise is positive and does not exceed (distance_matrix - 0.1)
%      noise_matrix = min(noise_matrix, distance_matrix - 0.1);
% 
%     % Step 4: Subtract the noise from the original distance matrix
%     noisy_distance_matrix = distance_matrix - noise_matrix;
% end




% function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon)
%     % Differential privacy Gaussian mechanism:
%     % sigma = sensitivity * sqrt(2*log(1.25/delta)) / epsilon
%     delta = 0.01;      % small constant; adjust as needed
%     sensitivity = 1;   % adjust if necessary for my work
%     sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
% 
%     % Generate Gaussian noise with mean 0 and standard deviation sigma
%     noise = sigma * randn(size(distance_matrix));
% 
%     % Add the noise to the original distance matrix
%     noisy_distance_matrix = distance_matrix + noise;
% 
%     % Optionally enforce non-negativity
%     noisy_distance_matrix = max(noisy_distance_matrix, 0.1);
% end





% % % BETTER
% function noisy_distance_matrix = add_noise_to_distance_matrix(distance_matrix, epsilon, noiseType)
%     % Normalize the distance matrix to [0,1]
%     d_min = min(distance_matrix(:));
%     d_max = max(distance_matrix(:));
%     normalized_distance = (distance_matrix - d_min) / (d_max - d_min);
% 
%     switch lower(noiseType)
%         case 'laplace'
%             % Laplace noise mechanism
%             scale = 1 / epsilon;
%             % Generate Laplace noise with the same size as the distance matrix
%             noise_matrix = laplace_noise(scale, size(distance_matrix));
%             % Scale the noise relative to the normalized distances
%             % This means that when the distance is large (normalized close to 1),
%             % the noise has a larger effect, and when the distance is small,
%             % the noise effect is diminished.
%             noise_matrix = noise_matrix .* normalized_distance;
% 
%             % Optional: Ensure the noise does not exceed (normalized_distance - 0.1)
%             noise_matrix = min(noise_matrix, normalized_distance - 0.1);
% 
%             % Subtract the noise from the normalized distance matrix
%             noisy_normalized_distance = normalized_distance - noise_matrix;
% 
%             % Rescale back to the original distance range
%             noisy_distance_matrix = noisy_normalized_distance * (d_max - d_min) + d_min;
% 
%         case 'gaussian'
%             % Differential privacy Gaussian mechanism
%             delta = 0.01;      % small constant; adjust as needed
%             sensitivity = 1;   % adjust if necessary for your application
%             sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
%             % Generate Gaussian noise with mean 0 and standard deviation sigma
%             noise = sigma * randn(size(distance_matrix));
%             % Scale the noise relative to the normalized distances
%             noise = noise .* normalized_distance;
%             % Add the noise to the normalized distance matrix
%             noisy_normalized_distance = normalized_distance + noise;
%             % Rescale back to the original distance range
%             noisy_distance_matrix = noisy_normalized_distance * (d_max - d_min) + d_min;
% 
%         otherwise
%             error('Unsupported noise type. Choose "laplace" or "gaussian".');
%     end
% end





