function agent = agentLPinit(agent, cost_matrix, NR_OBFLOC)
    
    NR_LOC_IN_AGENT = size(agent.node, 2);
    %% Formulate the linear programming problem

    cost_matrix_ = cost_matrix(agent.node', :); 
    agent.cost_vector = reshape(cost_matrix_', 1, NR_LOC_IN_AGENT*NR_OBFLOC); 

    A = full(agent.GeoI); 
    b = zeros(size(A, 1), 1); 

    Aeq = sparse(NR_LOC_IN_AGENT, NR_LOC_IN_AGENT*NR_OBFLOC); 
    beq = ones(NR_LOC_IN_AGENT, 1); 
    
    for k = 1:1:NR_LOC_IN_AGENT
        for l = 1:1:NR_OBFLOC
            Aeq(k, (k-1)*NR_OBFLOC + l) = 1;
        end
    end   
	lb = ones(NR_LOC_IN_AGENT*NR_OBFLOC, 1)*0.0000;
    ub = ones(NR_LOC_IN_AGENT*NR_OBFLOC, 1);
    options = optimoptions('linprog','Algorithm','dual-simplex');
    % cost_vector = abs(normrnd(3,10,[2500,1])); 
    z = linprog(agent.cost_vector, A, b, full(Aeq), beq, lb, ub, options);
    agent.decision = reshape(z, NR_OBFLOC, NR_LOC_IN_AGENT); 
end