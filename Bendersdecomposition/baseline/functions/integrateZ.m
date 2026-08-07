function Z = integrateZ(agent, env_parameters, obf_matrix)
    Z = obf_matrix; 
    NR_AGENT = size(agent, 2); 
    for i = 1:1:NR_AGENT
        if size(agent(i).decision, 1)*size(agent(i).decision, 2) > 0
            Z(agent(i).node_internal, :) = agent(i).decision; 
        end
    end
end