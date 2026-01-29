classdef MasterProgram
    properties
        cost_vector
        y
        w
        cut_set_A
        cut_set_b 
    end

    methods
        %% Constructor
        function this = MasterProgram()
            this.cut_set_A = []; 
            this.cut_set_b = []; 
        end

        %% Function: Calculate the cost vector
        function this = cost_vector_cal(this, subproblem, user, env_parameters)
            this.cost_vector = ones(1, size(env_parameters.longitude_selected,1)+size(subproblem, 1));
            for k = 1:1:size(env_parameters.longitude_selected, 1)
                for m = 1:1:size(user, 1)
                    for i = 1:1:user(m, 1).LR_loc_size
                        this.cost_vector(1, k) =  this.cost_vector(1, k) + subproblem(m, 1).Q_matrix(i,k)*user(m, 1).cost_matrix_whole(i,k)*exp(-user(m, 1).distance_matrix_LR2all(i,k)*env_parameters.EPSILON);
                    end
                end
            end
        end

        %% Function: Calculate the geo-obfuscation matrix
        function [this, cost, cost_exp] = calculate(this, env_parameters)
            lb = ones(size(this.cost_vector, 2), 1)*0.00001; 
            ub = ones(size(this.cost_vector, 2), 1)*1; 
            lb(1:size(env_parameters.longitude_selected, 1)) = 0.00001; 
            ub(1:size(env_parameters.longitude_selected, 1)) = 0.99;
            
            options = optimoptions('linprog','Algorithm','dual-simplex');
            % options = optimoptions('linprog', 'Algorithm', 'interior-point');
            [x, cost, exitflag, ~] = linprog(this.cost_vector, this.cut_set_A, this.cut_set_b, [], [], lb, ub, options); 
            if exitflag ~= 1
                options = optimoptions('linprog', 'Algorithm', 'interior-point');
                [x, cost, exitflag, ~] = linprog(this.cost_vector, this.cut_set_A, this.cut_set_b, [], [], lb, ub, options); 
            end
            if exitflag ~= 1
                x = zeros(size(this.cost_vector, 2), 1); 
                cost = 0; 
                cost_exp = 0;
            end
            this.y = x(1:size(env_parameters.longitude_selected, 1)); 
            cost_exp = this.cost_vector(1:size(env_parameters.longitude_selected, 1))*this.y; 
            this.w = x(size(env_parameters.longitude_selected, 1)+1:size(this.cost_vector, 2), 1); 
        end
        
        %% Function: Add new cuts from the subproblem 
        function this = add_newcuts(this, subproblem, user, m, env_parameters)
            new_cut_A_ = zeros(1, size(this.cost_vector, 2)); 
            new_cut_A_(user.obf_loc_ID) = subproblem.new_cut_A; 
            if subproblem.is_bounded == 1
                new_cut_A_(size(env_parameters.longitude_selected, 1)+m) = -1;
            end
            this.cut_set_A = [this.cut_set_A; new_cut_A_]; 
            this.cut_set_b = [this.cut_set_b; subproblem.new_cut_b]; 
            % subproblem.new_cut_A = [];
            % subproblem.new_cut_b = []; 
        end
    end
end

