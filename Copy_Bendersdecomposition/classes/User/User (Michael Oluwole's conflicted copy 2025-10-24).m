classdef User
    properties
        loc_ID
        longitude
        latitude 
        LR_loc_size
        LR_sample_size                                                      % Number of samples (100), ??Michael
        epsilon                                                             % Epsilon, ??Michael
        LR_loc_ID
        obf_range
        obf_loc_ID
        neighbor_threshold

        distance_matrix_LR_test
        distance_matrix_LR2all_test
        distance_matrix_LR2obf_test

        distance_matrix_LR                                                  % Distance between locally relevant locations
        distance_matrix_LR2all
        distance_matrix_LR2obf

        distance_matrix_LR_recovered
        distance_matrix_LR2obf_recovered
        cost_matrix_RL_recovered

        distance_matrix_LR_recovered_p
        distance_matrix_LR2obf_recovered_p
        cost_matrix_RL_recovered_p

        distance_matrix_LR_recovered_r
        distance_matrix_LR2obf_recovered_r
        cost_matrix_RL_recovered_r

        distance_matrix_LR_recovered_s
        distance_matrix_LR2obf_recovered_s
        cost_matrix_RL_recovered_s



        cost_matrix_upper
        cost_matrix_lower
        cost_matrix_whole
        cost_matrix_RL
        cost_matrix_RL_test


        obfuscation_matrix
        fitted_best_params_1                                                % ??Michael
        fitted_best_params_2                                                % ??Michael
        fitted_best_params_3                                                % ??Michael
        n_fitted_best_params_1                                              % ??
        n_fitted_best_params_2                                              % ??
        n_fitted_best_params_3                                              % ??
        % distance_matrix                                                     % ??
        best_pi
        
        fitted_best_coeffs_1                                                % ??Michael
        fitted_best_coeffs_2                                                % ??Michael
        fitted_best_coeffs_3                                                % ??Michael
        n_fitted_best_coeffs_1                                              % ??
        n_fitted_best_coeffs_2                                              % ??
        n_fitted_best_coeffs_3                                              % ??
        best_pi_poly

        fitted_best_w_1                                                     % ??Michael
        fitted_best_w_2                                                     % ??Michael
        fitted_best_w_3                                                     % ??Michael
        n_fitted_best_w_1                                                   % ??
        n_fitted_best_w_2                                                   % ??
        n_fitted_best_w_3                                                   % ??
        best_pi_rbf

        fitted_best_factors_1                                               % ??Michael
        fitted_best_factors_2                                               % ??Michael
        fitted_best_factors_3                                               % ??Michael
        fitted_best_factors_struct
        n_fitted_best_factors_1                                             % ??
        n_fitted_best_factors_2                                             % ??
        n_fitted_best_factors_3                                             % ??
        n_fitted_best_factors_struct 
        best_pi_svd
    end

    methods
        %% Constructor 
        function this = User(NODE_IDX, LR_LOC_SIZE, OBF_RANGE, NEIGHBOR_THRESHOLD, env_parameters)         
            this.loc_ID = NODE_IDX;
            this.longitude = env_parameters.longitude_selected(NODE_IDX); 
            this.latitude = env_parameters.latitude_selected(NODE_IDX);
            this.LR_loc_size = LR_LOC_SIZE;
            this.LR_sample_size = env_parameters.LR_SAMPLE_SIZE;                           % ??Michael
            this.epsilon = env_parameters.EPSILON;                                         % ??Michael
            this.LR_loc_ID = [];
            this.obf_range = OBF_RANGE; 
            this.neighbor_threshold = NEIGHBOR_THRESHOLD; 
            this.distance_matrix_LR = zeros(LR_LOC_SIZE, LR_LOC_SIZE);
            this.cost_matrix_upper = [];
            this.cost_matrix_whole = [];
        end

        %% Function: Initialize the users' properties
        function this = initialization(this, env_parameters)
            this = this.LR_identifier(env_parameters);                      % Identify the LR location IDs
            this = this.distance_matrix_LR_cal(env_parameters);             % Calculate the distance between LR locations
            this = this.obf_identifier(env_parameters);                     % Identify the obfuscated location IDs
            this = this.distance_matrix_LR2all_cal(env_parameters);         % Calculate the distance from LR locations to all the locations
            this = this.distance_matrix_LR2obf_cal();                       % Calculate the distance from LR locations to obfuscated locations
            this = this.cost_matrix_whole_cal(env_parameters); 
            this.cost_matrix_RL = this.cost_matrix_whole(:, this.obf_loc_ID); 
            % this = this.cost_matrix_cal(env_parameters);                    % Calculate the cost matrix

            %% Matrix approximation 
            this = this.gaussian_fit(env_parameters);                       % Calculate the 2D gaussian fit, ??Michael
            this = this.polynomial_fit(env_parameters);                     % Calculate the 2D polynomial fit, ??Michael
            this = this.RBF_fit(env_parameters);                            % Calculate the RBF fit, ??Michael
            this = this.lowrank_SVD_fit(env_parameters);                    % Calculate the Low Rank SVD fit, ??Michael    


            this = this.gaussian_noisy_parameters(env_parameters);                   % Calculate noisy parameters, ??Michael 
            this = this.polynomial_noisy_parameters(env_parameters); 
            this = this.RBF_noisy_parameters(env_parameters); 
            this = this.lowrank_SVD_noisy_parameters(env_parameters); 

            this = gaussian_recover(this, env_parameters); 
            this = polynomial_recover(this, env_parameters);
            this = RBF_recover(this, env_parameters);
            this = lowrank_SVD_recover(this, env_parameters);

            this.distance_matrix_LR_test = this.distance_matrix_LR;
            this.distance_matrix_LR2all_test = this.distance_matrix_LR2all;
            this.distance_matrix_LR2obf_test = this.distance_matrix_LR2obf;
            this.cost_matrix_RL_test = this.cost_matrix_RL; 
        end 

        %% Function:
        % function this = apply_svd(this)
        %     this.distance_matrix_LR = this.distance_matrix_LR_recovered_s; 
        %     this.distance_matrix_LR2obf = this.distance_matrix_LR2obf_recovered_s; 
        %     this.cost_matrix_RL = this.cost_matrix_RL_recovered_s;
        % end
        
        %% Function: Identify the local relevant location
        function this = LR_identifier(this, env_parameters)                             
            [~, distance] = shortestpathtree(env_parameters.G_mDP, this.loc_ID);            
            LR_ID_range = find(distance < env_parameters.GAMMA);                
            LR_loc_ID_ = randsample(size(LR_ID_range, 2), this.LR_loc_size); % Sample the set of relevant locations
            % LR_loc_ID_ = 1:1:this.LR_loc_size; 
            this.LR_loc_ID = LR_ID_range(LR_loc_ID_); 
        end

        %% Function: Calculate the distance matrix of the local relevant locations
        function this = distance_matrix_LR_cal(this, env_parameters)
            for i = 1:1:this.LR_loc_size
                for j = 1:1:this.LR_loc_size
                    % [~, path_distance] = shortestpath(env_parameters.G_mDP, s, t); 
                    % this.distance_matrix_LR(s, t) = path_distance; 
                    loc_i = [env_parameters.longitude_selected(this.LR_loc_ID(1, i), 1), env_parameters.latitude_selected(this.LR_loc_ID(1, i), 1)]; 
                    loc_j = [env_parameters.longitude_selected(this.LR_loc_ID(1, j), 1), env_parameters.latitude_selected(this.LR_loc_ID(1, j), 1)];
                    [this.distance_matrix_LR(i,j), ~, ~] = haversine(loc_i, loc_j); % Calculate the Haversine distance between locations
                end
            end
        end

        %% Function: Calculate the distance matrix from the local relevant locations to all the locations
        function this = distance_matrix_LR2all_cal(this, env_parameters)
            this.distance_matrix_LR2all = zeros(this.LR_loc_size, size(env_parameters.longitude_selected, 1)); 
            for i = 1:1:this.LR_loc_size
                for j = 1:1:size(env_parameters.longitude_selected, 1)
                    loc_i = [env_parameters.longitude_selected(this.LR_loc_ID(i), 1), env_parameters.latitude_selected(this.LR_loc_ID(i), 1)]; 
                    loc_j = [env_parameters.longitude_selected(j, 1), env_parameters.latitude_selected(j, 1)];
                    [this.distance_matrix_LR2all(i,j), ~, ~] = haversine(loc_i, loc_j); % Calculate the Haversine distance between locations
                end
            end
        end


        %% Function: Calculate the distance matrix from the local relevant locations to the obfuscated locations
        function this = distance_matrix_LR2obf_cal(this)
            this.distance_matrix_LR2obf = this.distance_matrix_LR2all(:, this.obf_loc_ID); 
        end

        %% Function: Identify the obfuscated locations given the obfuscation range (??Michael: Please change this function)
        function this = obf_identifier(this, env_parameters)
            this.obf_loc_ID = [];
            distances = [];
            for i = 1:1:size(env_parameters.longitude_selected, 1)
                loc_i = [env_parameters.longitude_selected(i, 1), env_parameters.latitude_selected(i, 1)];
                [distance_inst, ~, ~] = haversine(loc_i, [this.longitude, this.latitude]);
                if distance_inst < this.obf_range
                    this.obf_loc_ID = [this.obf_loc_ID, i];
                    distances = [distances, distance_inst];
                end
            end

            target_count = ceil(0.005 * env_parameters.nr_loc_selected);     
            if numel(this.obf_loc_ID) > target_count
                [~, sort_idx] = sort(distances, 'ascend');
                this.obf_loc_ID = this.obf_loc_ID(sort_idx(1:target_count));
            end
            % plot(env_parameters.longitude_selected(this.obf_loc_ID, 1), env_parameters.latitude_selected(this.obf_loc_ID, 1), 'o'); 
            % hold on; 
            % plot(this.longitude, this.latitude, '*'); 
        end


        %% Function: Calculate the cost matrix 
        function this = cost_matrix_cal(this, cr_table, env_parameters)
            this.cost_matrix_upper = zeros(this.LR_loc_size, size(this.obf_loc_ID, 2)); 
            
            %%%%%%%%%% This part will need to be modified using the cost reference table
            for i = 1:1:this.LR_loc_size
                for j = 1:1:size(this.obf_loc_ID, 2)
                    real_loc = [env_parameters.longitude_selected(this.LR_loc_ID(i)), env_parameters.latitude_selected(this.LR_loc_ID(i))]; 
                    obf_loc = [env_parameters.longitude_selected(this.obf_loc_ID(j)), env_parameters.latitude_selected(this.obf_loc_ID(j))]; 
                    for l = 1:1:size(cr_table.loc, 1)
                        real_distance(l) = haversine(real_loc, cr_table.loc(l, :)); 
                        obf_distance(l) = haversine(obf_loc, cr_table.loc(l, :));
                    end
                    [real_distance, real_idx] = min(real_distance); 
                    [obf_distance, obf_idx] = min(obf_distance); 
                    this.cost_matrix_upper(i, j) = cr_table.approximated_cost(real_idx, obf_idx)+real_distance+obf_distance; 
                    this.cost_matrix_lower(i, j) = cr_table.approximated_cost(real_idx, obf_idx)-real_distance-obf_distance;
                    this.cost_matrix_upper(i, j) = min([this.cost_matrix_upper(i, j), 999999]); 
                    this.cost_matrix_lower(i, j) = min([this.cost_matrix_lower(i, j), 999999]);
                end
            end
        end

        %% Function: Calculate the cost matrix 
        function this = cost_matrix_whole_cal(this, env_parameters)
            this.cost_matrix_upper = zeros(this.LR_loc_size, size(this.obf_loc_ID, 2)); 
            
            %%%%%%%%%% This part will need to be modified using the cost reference table
            for i = 1:1:this.LR_loc_size
                for j = 1:1:size(this.obf_loc_ID, 1)
                    [~, distance] = shortestpathtree(env_parameters.G, this.LR_loc_ID(i));
                    this.cost_matrix_whole(i, :) = min([999999*ones(size(distance)); distance]);
                end
            end
        end

        %% Function 1: Input: original matrix. Output: Parameters of fitted functions
        function this = gaussian_fit(this, env_parameters)
            A1 = this.distance_matrix_LR;          % LR x LR
            A2 = this.distance_matrix_LR2obf;      % LR x |obf|
            A3 = this.cost_matrix_RL;              % LR x |all|

            % Preallocate cells
            % this.fitted_best_params_1 = cell(this.LR_loc_size,1);
            % this.fitted_best_params_2 = cell(this.LR_loc_size,1);
            % this.fitted_best_params_3 = cell(this.LR_loc_size,1);

            best_params_1 = cell(this.LR_loc_size,1);
            best_params_2 = cell(this.LR_loc_size,1);
            best_params_3 = cell(this.LR_loc_size,1);

            % NOTE: I'm assuming reorder_fit_gaussians returns `best_params`
            % as a 1x12 vector. Adjust if your real function differs.
            [best_pi_inst, best_params, ~, ~, ~] = reorder_fit_gaussians(A1, A2, A3, 1.0, 1.0);
            best_params_1 = best_params(1, 1:6);
            best_params_2 = best_params(1, 7:12);
            best_params_3 = best_params(1, 13:18);
            
            % Normalize

            % M1 = vertcat(best_params_1{:});
            % M2 = vertcat(best_params_2{:});
            % M3 = vertcat(best_params_3{:});

            % M1_norm = normalize_rows(best_params_1);
            % M2_norm = normalize_rows(best_params_2);
            % M3_norm = normalize_rows(best_params_3);
            % 
            % this.fitted_best_params_1 = mat2cell(M1_norm, ones(size(M1_norm,1),1), size(M1_norm,2));
            % this.fitted_best_params_2 = mat2cell(M2_norm, ones(size(M2_norm,1),1), size(M2_norm,2));
            % this.fitted_best_params_3 = mat2cell(M3_norm, ones(size(M3_norm,1),1), size(M3_norm,2));
            this.fitted_best_params_1 = best_params_1;
            this.fitted_best_params_2 = best_params_2;
            this.fitted_best_params_3 = best_params_3;
            this.best_pi = best_pi_inst; 
        end

        %% Function 1: Input: original matrix. Output: Parameters of fitted functions
        function this = polynomial_fit(this, env_parameters)
            A1 = this.distance_matrix_LR;          % LR x LR
            A2 = this.distance_matrix_LR2obf;      % LR x |obf|
            A3 = this.cost_matrix_RL;              % LR x |all|
            % deg = 3;                             % or 2, 4, …

            % Preallocate cells
            this.fitted_best_coeffs_1 = cell(this.LR_loc_size,1);
            this.fitted_best_coeffs_2 = cell(this.LR_loc_size,1);
            this.fitted_best_coeffs_3 = cell(this.LR_loc_size,1);

            best_coeffs_1 = cell(this.LR_loc_size,1);
            best_coeffs_2 = cell(this.LR_loc_size,1);
            best_coeffs_3 = cell(this.LR_loc_size,1);


            [best_pi_inst, best_coeffs, ~, ~, ~] = reorder_fit_polynomials( ...
                                          A1, A2, A3, env_parameters.deg, 1.0, 1.0);

            best_coeffs_1 = best_coeffs(1, 1:10);
            best_coeffs_2 = best_coeffs(1, 11:20);
            best_coeffs_3 = best_coeffs(1, 21:30);

            % % Normalize
            % M1 = vertcat(best_coeffs_1{:});
            % M2 = vertcat(best_coeffs_2{:});
            % M3 = vertcat(best_coeffs_3{:});
            % 
            % M1_norm = normalize_rows(M1);
            % M2_norm = normalize_rows(M2);
            % M3_norm = normalize_rows(M3);

            % this.fitted_best_coeffs_1 = mat2cell(M1_norm, ones(size(M1_norm,1),1), size(M1_norm,2));
            % this.fitted_best_coeffs_2 = mat2cell(M2_norm, ones(size(M2_norm,1),1), size(M2_norm,2));
            % this.fitted_best_coeffs_3 = mat2cell(M3_norm, ones(size(M3_norm,1),1), size(M3_norm,2));
            this.fitted_best_coeffs_1 = best_coeffs_1;
            this.fitted_best_coeffs_2 = best_coeffs_2;
            this.fitted_best_coeffs_3 = best_coeffs_3;
            this.best_pi_poly = best_pi_inst;
        end

        %% Function 1: Input: original matrix. Output: Parameters of fitted functions
        function this = RBF_fit(this, env_parameters)
            A1 = this.distance_matrix_LR;                   % LR x LR
            A2 = this.distance_matrix_LR2obf;               % LR x |obf|
            A3 = this.cost_matrix_RL;                       % LR x |all|
            % NUM_CENTRES = 25;                             % try 9, 16, 25 … larger ⇒ more flexible, slower
            n = size(this.distance_matrix_LR, 1);        % first A1 gives n
            SIGMA = 0.35*n;         

            % Preallocate cells
            this.fitted_best_w_1 = cell(this.LR_loc_size,1);
            this.fitted_best_w_2 = cell(this.LR_loc_size,1);
            this.fitted_best_w_3 = cell(this.LR_loc_size,1);

            best_w_1 = cell(this.LR_loc_size,1);
            best_w_2 = cell(this.LR_loc_size,1);
            best_w_3 = cell(this.LR_loc_size,1);


            [best_pi_inst, best_w, ~, ~, ~] = reorder_fit_rbfs( ...
                                          A1, A2, A3, env_parameters.NUM_CENTRES, SIGMA, 1.0, 1.0);

            best_w_1 = best_w(1:env_parameters.NUM_CENTRES);
            best_w_2 = best_w(env_parameters.NUM_CENTRES+1:2*env_parameters.NUM_CENTRES);
            best_w_3 = best_w(2*env_parameters.NUM_CENTRES+1:end);

            % % Normalize
            % M1 = vertcat(best_w_1{:});
            % M2 = vertcat(best_w_2{:});
            % M3 = vertcat(best_w_3{:});

            % M1_norm = normalize_rows(M1);
            % M2_norm = normalize_rows(M2);
            % M3_norm = normalize_rows(M3);

            % this.fitted_best_w_1 = mat2cell(M1_norm, ones(size(M1_norm,1),1), size(M1_norm,2));
            % this.fitted_best_w_2 = mat2cell(M2_norm, ones(size(M2_norm,1),1), size(M2_norm,2));
            % this.fitted_best_w_3 = mat2cell(M3_norm, ones(size(M3_norm,1),1), size(M3_norm,2));
            this.fitted_best_w_1 = best_w_1;
            this.fitted_best_w_2 = best_w_2;
            this.fitted_best_w_3 = best_w_3;
            this.best_pi_rbf = best_pi_inst;
        end

        %% Function 1: Input: original matrix. Output: Parameters of fitted functions
        function this = lowrank_SVD_fit(this, env_parameters)
            A1 = this.distance_matrix_LR;          % LR x LR
            A2 = this.distance_matrix_LR2obf;      % LR x |obf|
            A3 = this.cost_matrix_RL;              % LR x |all|
            % rank_r   = 5;                        % tweak 3–8 for your grid size

            % Preallocate cells
            this.fitted_best_factors_1 = cell(this.LR_loc_size,1);
            this.fitted_best_factors_2 = cell(this.LR_loc_size,1);
            this.fitted_best_factors_3 = cell(this.LR_loc_size,1);

            best_factors_1 = cell(this.LR_loc_size,1);
            best_factors_2 = cell(this.LR_loc_size,1);
            best_factors_3 = cell(this.LR_loc_size,1);


            [best_pi_inst, best_factors, ~, ~, ~] = reorder_fit_lowrank_svd( ...
                                          A1, A2, A3, env_parameters.rank_r, 1.0, 1.0);

            % keep the full factors (U,S,V) for later recovery
            this.fitted_best_factors_struct = best_factors;

            best_factors_1 = diag(best_factors.S1)';    % 1×r
            best_factors_2 = diag(best_factors.S2)';    % 1×r
            best_factors_3 = diag(best_factors.S3)';    % 1×r

            % % Normalize
            % M1 = vertcat(best_factors_1{:});
            % M2 = vertcat(best_factors_2{:});
            % M3 = vertcat(best_factors_3{:});

            % M1_norm = normalize_rows(M1);
            % M2_norm = normalize_rows(M2);
            % M3_norm = normalize_rows(M3);

            % this.fitted_best_factors_1 = mat2cell(M1_norm, ones(size(M1_norm,1),1), size(M1_norm,2));
            % this.fitted_best_factors_2 = mat2cell(M2_norm, ones(size(M2_norm,1),1), size(M2_norm,2));
            % this.fitted_best_factors_3 = mat2cell(M3_norm, ones(size(M3_norm,1),1), size(M3_norm,2));
            this.fitted_best_factors_1 = best_factors_1;
            this.fitted_best_factors_2 = best_factors_2;
            this.fitted_best_factors_3 = best_factors_3;
            this.best_pi_svd = best_pi_inst;
        end

        %% Function 2: Input: parameters of fitted functions. Output: noisy parameters
        function this = gaussian_noisy_parameters(this, env_parameters)
            this.n_fitted_best_params_1 = add_noise_to_distance_matrix(this.fitted_best_params_1, this.epsilon, 'params');
            this.n_fitted_best_params_2 = add_noise_to_distance_matrix(this.fitted_best_params_2, this.epsilon, 'params');
            this.n_fitted_best_params_3 = add_noise_to_distance_matrix(this.fitted_best_params_3, this.epsilon, 'params');
        end

        function this = polynomial_noisy_parameters(this, env_parameters)
            this.n_fitted_best_coeffs_1 = add_noise_to_distance_matrix(this.fitted_best_coeffs_1, this.epsilon, 'params');
            this.n_fitted_best_coeffs_2 = add_noise_to_distance_matrix(this.fitted_best_coeffs_2, this.epsilon, 'params');
            this.n_fitted_best_coeffs_3 = add_noise_to_distance_matrix(this.fitted_best_coeffs_3, this.epsilon, 'params');
        end

        function this = RBF_noisy_parameters(this, env_parameters)
            this.n_fitted_best_w_1 = add_noise_to_distance_matrix(this.fitted_best_w_1, this.epsilon, 'param');
            this.n_fitted_best_w_2 = add_noise_to_distance_matrix(this.fitted_best_w_2, this.epsilon, 'param');
            this.n_fitted_best_w_3 = add_noise_to_distance_matrix(this.fitted_best_w_3, this.epsilon, 'param');
        end

        function this = lowrank_SVD_noisy_parameters(this, env_parameters)
            this.n_fitted_best_factors_1 = add_noise_to_distance_matrix(this.fitted_best_factors_1, this.epsilon, 'param');
            this.n_fitted_best_factors_2 = add_noise_to_distance_matrix(this.fitted_best_factors_2, this.epsilon, 'param');
            this.n_fitted_best_factors_3 = add_noise_to_distance_matrix(this.fitted_best_factors_3, this.epsilon, 'param');

            fac_clean = this.fitted_best_factors_struct;   % U,S,V from fitting
            fac_noisy = fac_clean;                         % start as a copy
        
            % ------- A1 (n×n) -------
            v1 = this.n_fitted_best_factors_1(:);  % vectorized noisy singular values
            % conform rank to U1/V1 in case of length mismatch
            r1 = min([numel(v1), size(fac_clean.U1,2), size(fac_clean.V1,2)]);
            if r1 < 1
                error('noisy_parameters:RankMismatchA1', 'Computed rank r1 < 1 for A1.');
            end
            fac_noisy.U1 = fac_clean.U1(:, 1:r1);
            fac_noisy.V1 = fac_clean.V1(:, 1:r1);
            fac_noisy.S1 = diag(v1(1:r1));
        
            % ------- A2 (n×m) -------
            v2 = this.n_fitted_best_factors_2(:);
            r2 = min([numel(v2), size(fac_clean.U2,2), size(fac_clean.V2,2)]);
            if r2 < 1
                error('noisy_parameters:RankMismatchA2', 'Computed rank r2 < 1 for A2.');
            end
            fac_noisy.U2 = fac_clean.U2(:, 1:r2);
            fac_noisy.V2 = fac_clean.V2(:, 1:r2);
            fac_noisy.S2 = diag(v2(1:r2));
        
            % ------- A3 (n×m) -------
            v3 = this.n_fitted_best_factors_3(:);
            r3 = min([numel(v3), size(fac_clean.U3,2), size(fac_clean.V3,2)]);
            if r3 < 1
                error('noisy_parameters:RankMismatchA3', 'Computed rank r3 < 1 for A3.');
            end
            fac_noisy.U3 = fac_clean.U3(:, 1:r3);
            fac_noisy.V3 = fac_clean.V3(:, 1:r3);
            fac_noisy.S3 = diag(v3(1:r3));

            tau1 = 0.5/this.epsilon;  % rough; tune on validation
            tau2 = 0.5/this.epsilon;
            tau3 = 0.5/this.epsilon;
            
            s1 = diag(fac_noisy.S1); s1 = max(s1 - tau1, 0); fac_noisy.S1 = diag(s1);
            s2 = diag(fac_noisy.S2); s2 = max(s2 - tau2, 0); fac_noisy.S2 = diag(s2);
            s3 = diag(fac_noisy.S3); s3 = max(s3 - tau3, 0); fac_noisy.S3 = diag(s3);
        
            % store for recover()
            this.n_fitted_best_factors_struct = fac_noisy;
        end

        function Sshr = shrink_singular_values(S, tau)
            s = diag(S);
            s = max(s - tau, 0);  % soft-threshold
            Sshr = diag(s);
        end

        %%
        function this = gaussian_recover(this, env_parameters)
            N = size(this.distance_matrix_LR, 1);
            M = size(this.distance_matrix_LR2obf, 2);
            best_params  = [this.n_fitted_best_params_1 this.n_fitted_best_params_2 this.n_fitted_best_params_3]; 

            % mu = this.fitted_best_params_1(1, 1:2); 
            % mu = mu'; 
            % Sigma = this.fitted_best_params_1(1, 3:6);
            % Sigma = reshape(Sigma, 2, 2); 
            % this.distance_matrix_LR = generate2DGaussianMatrix(N, M, mu, Sigma);
            % [A1_hat, ~, ~] = recover_gaussian_approximations(N, N, best_params, this.best_pi);  

            [A1_hat, A2_hat, A3_hat] = recover_gaussian_approximations(N, M, best_params, this.best_pi);
            this.distance_matrix_LR_recovered     = A1_hat; 
            this.distance_matrix_LR2obf_recovered = A2_hat;
            this.cost_matrix_RL_recovered         = A3_hat;            
            violRatio_1 = violation_ratio(this.distance_matrix_LR, this.distance_matrix_LR_recovered); 
            violRatio_2 = violation_ratio(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered); 

            relErr_1 = relative_error(this.distance_matrix_LR, this.distance_matrix_LR_recovered);
            relErr_2 = relative_error(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered);
            relErr_3 = relative_error(this.cost_matrix_RL, this.cost_matrix_RL_recovered);
        end

        function this = polynomial_recover(this, env_parameters)
            N = size(this.distance_matrix_LR, 1);
            M = size(this.distance_matrix_LR2obf, 2);
            best_coeffs = [this.n_fitted_best_coeffs_1 this.n_fitted_best_coeffs_2 this.n_fitted_best_coeffs_3]; 

            [A1_hat_p, A2_hat_p, A3_hat_p] = recover_polynomial_approximations(N, M, best_coeffs, env_parameters.deg, this.best_pi_poly);
            A1_hat_p = max(A1_hat_p, 0);
            A1_hat_p = 0.5*(A1_hat_p + A1_hat_p');
            A1_hat_p(1:size(A1_hat_p,1)+1:end) = 0;
            A1_hat_p = project_metric(A1_hat_p);  % Floyd–Warshall projection

            this.distance_matrix_LR_recovered_p     = A1_hat_p; 
            % this.distance_matrix_LR2obf_recovered_p = A2_hat_p;
            % this.cost_matrix_RL_recovered_p         = A3_hat_p;   
            this.distance_matrix_LR2obf_recovered_p = project_rectangular_triangle_tighten(A1_hat_p, A2_hat_p, 2);
            this.cost_matrix_RL_recovered_p         = project_rectangular_triangle_tighten(A1_hat_p, A3_hat_p, 2);
            violRatio_1_p = violation_ratio(this.distance_matrix_LR, this.distance_matrix_LR_recovered_p); 
            violRatio_2_p = violation_ratio(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_p); 

            relErr_1_p = relative_error(this.distance_matrix_LR, this.distance_matrix_LR_recovered_p);
            relErr_2_p = relative_error(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_p);
            relErr_3_p = relative_error(this.cost_matrix_RL, this.cost_matrix_RL_recovered_p);
        end

        function this = RBF_recover(this, env_parameters)
            N = size(this.distance_matrix_LR, 1);
            M = size(this.distance_matrix_LR2obf, 2);
            best_w = [this.n_fitted_best_w_1 this.n_fitted_best_w_2 this.n_fitted_best_w_3]; 
            
            [A1_hat_r, A2_hat_r, A3_hat_r] = recover_rbfs_approximations(N, M, best_w, env_parameters.NUM_CENTRES, 0.35*N, this.best_pi_rbf);
            this.distance_matrix_LR_recovered_r     = A1_hat_r; 
            this.distance_matrix_LR2obf_recovered_r = A2_hat_r;
            this.cost_matrix_RL_recovered_r         = A3_hat_r;            
            violRatio_1_r = violation_ratio(this.distance_matrix_LR, this.distance_matrix_LR_recovered_r); 
            violRatio_2_r = violation_ratio(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_r); 

            relErr_1_r = relative_error(this.distance_matrix_LR, this.distance_matrix_LR_recovered_r);
            relErr_2_r = relative_error(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_r);
            relErr_3_r = relative_error(this.cost_matrix_RL, this.cost_matrix_RL_recovered_r);
        end
            
        function this = lowrank_SVD_recover(this, env_parameters)
            N = size(this.distance_matrix_LR, 1);
            M = size(this.distance_matrix_LR2obf, 2);
            best_factors = this.n_fitted_best_factors_struct; 

            [A1_hat_s, A2_hat_s, A3_hat_s] = recover_lowrank_svd_approximations(N, M, best_factors, this.best_pi_svd);
            A1_hat_s = max(A1_hat_s, 0);
            A1_hat_s = 0.5*(A1_hat_s + A1_hat_s');
            A1_hat_s(1:size(A1_hat_s,1)+1:end) = 0;
            A1_hat_s = project_metric(A1_hat_s);  % Floyd–Warshall projection

            this.distance_matrix_LR_recovered_s     = A1_hat_s;                       
            this.distance_matrix_LR2obf_recovered_s = project_rectangular_triangle_tighten(A1_hat_s, A2_hat_s, 2);
            this.cost_matrix_RL_recovered_s         = A3_hat_s;
            violRatio_1_s = violation_ratio(this.distance_matrix_LR, this.distance_matrix_LR_recovered_s); 
            violRatio_2_s = violation_ratio(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_s); 

            relErr_1_s = relative_error(this.distance_matrix_LR, this.distance_matrix_LR_recovered_s);
            relErr_2_s = relative_error(this.distance_matrix_LR2obf, this.distance_matrix_LR2obf_recovered_s);
            relErr_3_s = relative_error(this.cost_matrix_RL, this.cost_matrix_RL_recovered_s);
        end


        function G = generate2DGaussianMatrix(this, N, M, mu, Sigma)
            % generate2DGaussianMatrix generates an N-by-M matrix representing a 2D Gaussian distribution
            %
            % Inputs:
            %   N      - Number of rows (height)
            %   M      - Number of columns (width)
            %   mu     - 2x1 mean vector [mu_x; mu_y]
            %   Sigma  - 2x2 covariance matrix
            %
            % Output:
            %   G      - N-by-M matrix with Gaussian values

            % Create meshgrid centered around mu
            [X, Y] = meshgrid(1:M, 1:N);

            % Vectorize coordinates
            X = X - mu(1);
            Y = Y - mu(2);

            % Inverse and determinant of covariance
            Sigma_inv = inv(Sigma);
            denom = 2 * pi * sqrt(det(Sigma));

            % Compute Gaussian values
            G = zeros(N, M);
            for i = 1:N
                for j = 1:M
                    diff = [X(i,j); Y(i,j)];
                    G(i,j) = exp(-0.5 * diff' * Sigma_inv * diff);
                end
            end
        end
    end
end



