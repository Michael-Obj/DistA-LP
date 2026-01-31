classdef Subproblem
    properties
        z
        cost_vector
        cost_vector_lower
        geoI_matrix
        geoI_matrix_A
        geoI_matrix_B
        Unit_matrix
        Unit_matrix_A
        Unit_matrix_B
        geoI_b
        unit_b
        Q_matrix
        Q_matrix_obf
        new_cut_A
        new_cut_b
        is_bounded
        cost
        cost_test
        cost_current
        % cost_lower
    end

    methods
        %% Constructor: Initialize all the properties
        function this = Subproblem()
            this.z = [];
            this.cost_vector = [];
            this.geoI_matrix_A = [];
            this.geoI_matrix_B = [];
            this.Unit_matrix_A = [];
            this.Unit_matrix_B = [];
            this.geoI_b = [];
            this.unit_b = []; 
            this.Q_matrix = [];
            this.Q_matrix_obf = [];
            this.new_cut_A = []; 
            this.new_cut_b = []; 
            this.is_bounded = 0; 
            this.cost = 999; 
        end

        %% Calculate the Q matrix
        function this = Q_matrix_cal(this, user, exp_range)
            for i = 1:1:size(user.distance_matrix_LR2all, 1)
                for k = 1:1:size(user.distance_matrix_LR2all, 2)
                    if user.distance_matrix_LR2all(i, k) >= exp_range &&  user.distance_matrix_LR2all(i, k) <= user.obf_range        % If the distance is higher than the ExpMech range
                        this.Q_matrix(i, k) = 1;
                    else
                        this.Q_matrix(i, k) = 0;
                    end
                end
            end
            this.Q_matrix_obf = this.Q_matrix(:, user.obf_loc_ID); 
            this.Q_matrix_obf = reshape(this.Q_matrix_obf', 1, size(this.Q_matrix_obf, 1)*size(this.Q_matrix_obf, 2)); 
            this.geoI_b = zeros(user.LR_loc_size*(user.LR_loc_size-1)*size(user.obf_loc_ID, 2), 1); 
        end

        %% Calculate the Geo-Ind matrix, including GeoI_matrix, GeoI_matrix_A, GeoI_matrix_B, and geoI_b
        function this = geo_matrix_cal(this, user, env_parameters)
            idx = 1;
            this.geoI_matrix = sparse(user.LR_loc_size*(user.LR_loc_size-1)*size(user.obf_loc_ID, 2), user.LR_loc_size*size(user.obf_loc_ID, 2)); 
            this.geoI_matrix_A = sparse(user.LR_loc_size*(user.LR_loc_size-1)*size(user.obf_loc_ID, 2), user.LR_loc_size*size(user.obf_loc_ID, 2));
            this.geoI_matrix_B = sparse(user.LR_loc_size*(user.LR_loc_size-1)*size(user.obf_loc_ID, 2), user.LR_loc_size*size(user.obf_loc_ID, 2));
            for i = 1:1:user.LR_loc_size
                for j = 1:1:user.LR_loc_size
                    if i ~= j && user.distance_matrix_LR(i, j) <= user.neighbor_threshold
                        for k = 1:1:size(user.obf_loc_ID, 2)
                            this.geoI_matrix(idx, (i-1)*size(user.obf_loc_ID, 2) + k) = 1; 
                            this.geoI_matrix(idx, (j-1)*size(user.obf_loc_ID, 2) + k) = -exp(env_parameters.EPSILON*user.distance_matrix_LR(i,j)); 
                            this.geoI_matrix_A(idx, :) = this.geoI_matrix(idx, :).*(1-this.Q_matrix_obf);
                            this.geoI_matrix_B(idx, :) = this.geoI_matrix(idx, :).*this.Q_matrix_obf.*(exp(reshape(user.distance_matrix_LR2obf', 1, size(user.distance_matrix_LR2obf,1)*size(user.distance_matrix_LR2obf,2))));
                            idx = idx + 1;
                        end
                    end
                end
            end
            this.geoI_b = zeros(user.LR_loc_size*(user.LR_loc_size-1)*size(user.obf_loc_ID, 2), 1); 
        end

        %% Calculate the unit measure matrix
        function this = unit_matrix_cal(this, user)
            this.Unit_matrix = sparse(user.LR_loc_size, user.LR_loc_size*size(user.obf_loc_ID, 2)); 
            this.Unit_matrix_A = sparse(user.LR_loc_size, user.LR_loc_size*size(user.obf_loc_ID, 2));
            this.Unit_matrix_B = sparse(user.LR_loc_size, user.LR_loc_size*size(user.obf_loc_ID, 2));
            for i = 1:1:user.LR_loc_size
                for l = 1:1:size(user.obf_loc_ID, 2)
                    this.Unit_matrix(i, (i-1)*size(user.obf_loc_ID, 2) + l) = 1;
                end
                this.Unit_matrix_A(i, :) = this.Unit_matrix(i, :).*(1-this.Q_matrix_obf);
                this.Unit_matrix_B(i, :) = this.Unit_matrix(i, :).*this.Q_matrix_obf.*(exp(reshape(this.Q_matrix_obf', 1, size(this.Q_matrix_obf,1)*size(this.Q_matrix_obf,2))));
            end   
            this.unit_b = ones(user.LR_loc_size, 1);
        end

        %% Calculate the geo-obfuscation matrix
        function [this, user] = calculate(this, user, m, master_program, env_parameters)
            this.cost = 9999;
            y_ = master_program.y(user.obf_loc_ID); 

            y_transform = zeros(size(user.obf_loc_ID,2)*user.LR_loc_size, size(user.obf_loc_ID,2)); 
            for i = 1:1:user.LR_loc_size
                for k = 1:1:size(user.obf_loc_ID, 2)
                    y_transform((i-1)*size(user.obf_loc_ID, 2) + k, k) = exp(-env_parameters.EPSILON*user.distance_matrix_LR2obf(i,k)); 
                end
            end

            z_exp = y_transform*y_; 
            A = [-this.geoI_matrix_A; -this.Unit_matrix_A; this.Unit_matrix_A];
            B = [-this.geoI_matrix_B; -this.Unit_matrix_B; this.Unit_matrix_B]; 
            b = [-this.geoI_b; -this.unit_b; this.unit_b];
        
            
            % Only for testing
            % A_lower = [-this.Unit_matrix_A; this.Unit_matrix_A];
            % B_lower = [-this.Unit_matrix_B; this.Unit_matrix_B]; 
            % b_lower = [-this.unit_b; this.unit_b];

            % this.cost_vector = reshape(user.cost_matrix_upper', 1, size(user.cost_matrix_upper, 1)*size(user.cost_matrix_upper, 2)); 
            this.cost_vector = reshape(user.cost_matrix_RL', 1, size(user.cost_matrix_RL, 1)*size(user.cost_matrix_RL, 2)); 

            this.cost_vector_lower = reshape(user.cost_matrix_lower', 1, size(user.cost_matrix_lower, 1)*size(user.cost_matrix_lower, 2)); 
            c = this.cost_vector'; 
            c_lower = this.cost_vector_lower';


            lb = zeros(size(c)); 
            lb = ones(size(c))*0.01; 
            lb = lb'; 
            ub = [];
            ub = ones(size(c))*0.99;
            ub = ub'; 

            %% Primal problem
            % options = optimoptions('linprog','Algorithm','dual-simplex','Preprocess', 'none');
            options.MaxIterations = 1000000; 
            [this.z, fval, exitflag] = linprog(c, -A, -(b-B*z_exp), [], [], lb, ub, options);
            if exitflag == 1
                
                user.obfuscation_matrix = reshape(this.z', size(user.cost_matrix_upper, 2), size(user.cost_matrix_upper, 1)); 
                this.cost_test = sum(sum(user.obfuscation_matrix'.*user.cost_matrix_RL_test));
                this.cost_current = fval; 
                this.cost = fval;
            end

            % [this.z, this.cost_lower, exitflag] = linprog(c_lower, -A, -(b-B*z_exp), [], [], lb, ub, options);
            % if exitflag ~= 1
            %     this.cost_lower = 0; 
            % end

            %% Dual problem
            lb = zeros(size(A, 1), 1);
            ub = [];
            % options = optimoptions('linprog','Algorithm','dual-simplex','Preprocess', 'none');
            options.MaxIterations = 10000; 
            [u, fval, exitflag, ~, ~] = linprog(-(b-B*z_exp), A', c, [], [], lb, ub, options); 

            if exitflag == -3
                % options = optimoptions('linprog','Algorithm','dual-simplex','Preprocess', 'none');
                options.MaxIterations = 10000; 
                
                [u, ~, exitflag, ~, ~] = linprog([], A', zeros(size(c)), (b-B*z_exp)', 1, lb, ub, options);
                u = u/sqrt(sum(u.^2));
                if exitflag == 1
                    this.new_cut_A = -u'*B*y_transform; 
                    this.new_cut_b = -u'*b;
                else 
                    this.new_cut_A = zeros(1, size(y_transform, 2)); 
                    this.new_cut_b = 0;
                end
                this.is_bounded = 0; 
                this.cost = 999999;
                this.cost_current = 999999;
                this.cost_test = 999999; 
                
            else
                this.cost = -fval;
                this.cost_current = -fval;
                this.cost_test = -fval; 
                this.new_cut_A = zeros(1, size(y_transform, 2)); 
                this.new_cut_b = 0;
                if master_program.w(m) < -fval
                    this.new_cut_A = -u'*B*y_transform;
                    this.new_cut_b = -u'*b;
                    this.is_bounded = 1;           
                end
            end
        end
    end
end