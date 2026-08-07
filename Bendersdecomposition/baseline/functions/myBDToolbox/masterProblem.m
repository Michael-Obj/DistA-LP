function [masteragent, agent, lowerbound, is_cut] = masterProblem(masteragent, agent, env_parameters)
    env_parameters.NR_LOC_IN_AGENT = size(masteragent.node, 2);
    is_cut = 0; 
    %% Formulate the linear programming problem

    env_parameters.cost_matrix_ = env_parameters.cost_matrix(masteragent.node', :); 
    masteragent.cost_vector = reshape(env_parameters.cost_matrix_', 1, env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC); 
    masteragent.cost_vector = [masteragent.cost_vector, ones(1, env_parameters.NR_AGENT)]; 

    A = [masteragent.GeoI, zeros(size(masteragent.GeoI, 1), env_parameters.NR_AGENT)];
    A = [A; masteragent.cuts_A]; 
    b = [zeros(size(masteragent.GeoI, 1), 1); masteragent.cuts_b]; 

    Aeq = sparse(env_parameters.NR_LOC_IN_AGENT, env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT); 
    beq = ones(env_parameters.NR_LOC_IN_AGENT, 1); 
    
    for k = 1:1:env_parameters.NR_LOC_IN_AGENT
        for l = 1:1:env_parameters.NR_OBFLOC
            Aeq(k, (k-1)*env_parameters.NR_OBFLOC + l) = 1;
        end
    end   
	lb = ones(env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT, 1)*0.0001;
    ub = ones(env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT, 1);
    ub(env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+1:env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT, 1) = 9999; 
    options = optimoptions('linprog','Algorithm','dual-simplex');
    % cost_vector = abs(normrnd(3,10,[2500,1])); 
    [z, lowerbound, exitflag] = linprog(masteragent.cost_vector, A, b, Aeq, beq, lb, ub, options);
    %loss_boundary_rec=masteragent.cost_vector*z;
    lowerbound = lowerbound - 0.0001*sum(masteragent.cost_vector); 
    if exitflag == 1
        disp('Hahahaha ~')
        for i = 1:1:env_parameters.NR_AGENT
            agent(i).z = z(env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC+i, 1); 
        end
        masteragent.decision = reshape(z(1:env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC, 1), env_parameters.NR_OBFLOC, env_parameters.NR_LOC_IN_AGENT); 
        masteragent.decision = masteragent.decision';

        is_cut = 1; 
    else
        disp('Sorry ...')
        % masteragent.cuts_A(size( masteragent.cuts_A, 1), :) = [];
        % masteragent.cuts_b(size( masteragent.cuts_b, 1), :) = [];
        % A = [masteragent.GeoI; masteragent.cuts_A]; 
        % b = [zeros(size(masteragent.GeoI, 1), 1); masteragent.cuts_b]; 
        % [z, lowerbound, exitflag] = linprog(masteragent.cost_vector, A, b, Aeq, beq, lb, ub, options);
        % masteragent.decision = reshape(z(1:env_parameters.NR_LOC_IN_AGENT*env_parameters.NR_OBFLOC, 1), env_parameters.NR_OBFLOC, env_parameters.NR_LOC_IN_AGENT); 
        % masteragent.decision = masteragent.decision';
    end
    
end