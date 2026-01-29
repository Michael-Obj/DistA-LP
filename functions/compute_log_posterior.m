% function [posterior_prob, PL_matrix, PL_max] = compute_log_posterior(raw_cells, noisy_cells, distance_matrix, num_locations, b_value, cardinality_N)
%     % raw_cells, noisy_cells: 1 x num_locations cell arrays of numeric matrices
%     posterior_prob = zeros(1, num_locations);
%     conditional_probability_log = zeros(1, num_locations);
% 
%     for loc_idx = 1:num_locations
%         raw_M   = raw_cells{loc_idx};
%         noisy_M = noisy_cells{loc_idx};
% 
%         % Basic shape check (optional)
%         if ~isequal(size(raw_M), size(noisy_M))
%             error('Size mismatch for location %d: raw %s vs noisy %s', ...
%                   loc_idx, mat2str(size(raw_M)), mat2str(size(noisy_M)));
%         end
% 
%         % Accumulate log-likelihood under Laplace(b_value)
%         % constant term per entry + distance-dependent term
%         const_log = cardinality_N * log(1 / (2*b_value));
%         acc = 0;
%         for i = 1:size(raw_M, 1)
%             for j = 1:size(raw_M, 2)
%                 diff_ij = abs(noisy_M(i, j) - raw_M(i, j));
%                 acc = acc + const_log - diff_ij / b_value;
%             end
%         end
%         conditional_probability_log(loc_idx) = acc;
%     end
% 
%     % We keep log-posteriors (unnormalized) as your code intended
%     posterior_prob = conditional_probability_log;
% 
%     % Posterior Leakage matrix
%     PL_matrix = zeros(num_locations, num_locations);
%     for i = 1:num_locations
%         for j = 1:num_locations
%             if i ~= j && distance_matrix(i, j) ~= 0
%                 PL_matrix(i, j) = abs(posterior_prob(i) - posterior_prob(j)) / distance_matrix(i, j);
%             end
%         end
%     end
% 
%     PL_max = max(PL_matrix(:));
% end



% Function to compute posterior probabilities
function [posterior_prob, PL_matrix, PL_max] = compute_log_posterior(raw_distance_matrix, noisy_distance_matrix, distance_matrix, num_locations, b_value, cardinality_N)
    % Initialize posterior probabilities matrix
    posterior_prob = zeros(1, size(raw_distance_matrix, 1));

    % Uniform prior probability for each location
    prior_prob = 1 / num_locations;

    conditional_probability = ones(1, num_locations);
    conditional_probability_log = zeros(1, num_locations);


    % Compute the posterior probability for each element in the matrix
    for loc_idx = 1:1:num_locations
        raw_distance_matrix_ = raw_distance_matrix{loc_idx};
        for i = 1:size(raw_distance_matrix_, 1)
            for j = 1:size(raw_distance_matrix_, 2)
                diff = abs(noisy_distance_matrix(i, j) - raw_distance_matrix_(i, j));
                conditional_probability(1, loc_idx) = conditional_probability(1, loc_idx)*(1 / (2 * b_value))^cardinality_N * exp(-diff / b_value);
                conditional_probability_log(1, loc_idx) = conditional_probability_log(1, loc_idx) + log((1 / (2 * b_value))^cardinality_N * exp(-diff / b_value)); 
                % sum_con_prob = sum(conditional_probability);
            end
        end
    end

    % % Compute normalization constant using the log-sum-exp trick:
    % max_log = max(conditional_probability_log);
    % log_sum_exp = max_log + log(sum(exp(conditional_probability_log - max_log)));
    % 
    % % Now, compute normalized log posterior probabilities:
    % normalized_log_posterior = conditional_probability_log - log_sum_exp;
    % 
    % % If you need probabilities in the usual [0,1] range, exponentiate:
    % posterior_prob = exp(normalized_log_posterior);

    for loc_idx = 1:1:num_locations
        % posterior_prob(1, loc_idx) = conditional_probability(1, loc_idx)/sum(conditional_probability); 
        posterior_prob(1, loc_idx) = conditional_probability_log(1, loc_idx);  
    end


    %% Create the Posterior Leakage matrix
    PL_matrix = zeros(num_locations, num_locations); 
    for i = 1:1:num_locations
        for j = 1:1:num_locations
            if i ~= j
                PL_matrix(i,j) = abs(posterior_prob(1, i) - posterior_prob(1, j))/distance_matrix(i, j); 
            end
        end
    end


    %% Calculate the max PL
    PL_max = max(max(PL_matrix)); 
end

















%%
% % % Function to compute posterior probabilities
% function posterior_prob = compute_log_posterior(raw_distance_matrix, noisy_distance_matrix, num_locations, b_value, cardinality_N)
%     % Initialize posterior probabilities matrix
%     posterior_prob = zeros(size(raw_distance_matrix));
% 
%     % Uniform prior probability for each location
%     prior_prob = 1 / num_locations;
% 
%     conditional_probability = ones(1, num_locations);
%     conditional_probability_log = zeros(1, num_locations);
% 
%     % Compute the posterior probability for each element in the matrix
%     for loc_idx = 1:1:num_locations
%         raw_distance_matrix_ = raw_distance_matrix;
%         for i = 1:size(raw_distance_matrix_, 1)
%             for j = 1:size(raw_distance_matrix_, 2)
%                 diff = abs(noisy_distance_matrix(i, j) - raw_distance_matrix_(i, j));
%                 conditional_probability(1, loc_idx) = conditional_probability(1, loc_idx)*(1 / (2 * b_value))^cardinality_N * exp(-diff / b_value);
%                 conditional_probability_log(1, loc_idx) = conditional_probability_log(1, loc_idx) + log((1 / (2 * b_value))^cardinality_N * exp(-diff / b_value)); 
%                 sum_con_prob = sum(conditional_probability);
%             end
%         end
%     end
% 
%     for loc_idx = 1:1:num_locations
%         posterior_prob(1, loc_idx) = conditional_probability(1, loc_idx)/sum(conditional_probability); 
%     end
% end




% Function to compute the posterior probability for each location
% function posterior_prob = compute_log_posterior(perturbation_prob, num_locations)
%     posterior_prob = zeros(size(perturbation_prob));
% 
%     % Compute the sum of all perturbation probabilities weighted by priors
%     prior_probability = 1 / num_locations;
%     conditional_probability = zeros(1, num_locations);
% 
%     for loc_idx = 1:num_locations
%         % P(D_i | i) * P(i) for each location
%         for i = 1:size(perturbation_prob, 1)
%             for j = 1:size(perturbation_prob, 2)
%                 conditional_probability(1, loc_idx) = conditional_probability(1, loc_idx) + log(perturbation_prob(i, j) * prior_probability);
%                 con = sum(conditional_probability);
%             end
%         end
%     end
% 
%     for loc_idx = 1:1:num_locations
%         posterior_prob(1, loc_idx) = conditional_probability(1, loc_idx)/con; 
%     end
% 
%     % Calculate the posterior probability for the current location
%     % posterior_prob = (perturbation_prob(location_idx) * prior_probability) / sum(conditional_probability(1, i));
% 
%     % % Compute the sum of all conditional probabilities for normalization
%     % normalization_factor = sum(conditional_probability);
%     % 
%     % % Calculate the posterior probabilities for all locations
%     % posterior_prob = conditional_probability / normalization_factor;
% end



 
% % Function to compute posterior probabilities
% function posterior_prob = compute_log_posterior(perturbation_prob, num_locations)
%     % Initialize posterior probabilities matrix
%     posterior_prob = zeros(size(perturbation_prob));
% 
%     % Uniform prior probability for each location
%     prior_prob = 1 / num_locations;
% 
%     conditional_probability = ones(1, num_locations);
%     conditional_probability_log = zeros(1, num_locations);
% 
%     % Compute the posterior probability for each element in the matrix
%     for loc_idx = 1:1:num_locations
%         % perturbation_prob_ = perturbation_prob{loc_idx};
%         for i = 1:size(perturbation_prob, 1)
%             for j = 1:size(perturbation_prob, 2)
%                 conditional_probability(1, loc_idx) = conditional_probability(1, loc_idx)*perturbation_prob(i, j);
%                 conditional_probability_log(1, loc_idx) = conditional_probability_log(1, loc_idx) + log(perturbation_prob(i, j)); 
%             end
%         end
%     end
%     for loc_idx = 1:1:num_locations
%         posterior_prob(1, loc_idx) = conditional_probability(1, loc_idx)/sum(conditional_probability); 
%     end
% end



