function masteragent = addNewCuts(masteragent, agent, env_parameters)
    new_cut_A_ = []; 
    new_cut_b_ = []; 
    for i = 1:1:env_parameters.NR_AGENT 
                new_cut_A__ = sparse(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT);
                if agent(i).isunbounded == 1  
                    for j = 1:1:size(agent(i).node_boundary, 2)
                        node_j = find(masteragent.node_boundary == agent(i).node_boundary(j)); 
                        for k = 1:1:env_parameters.NR_OBFLOC
                            new_cut_A__(1, (node_j-1)*env_parameters.NR_OBFLOC+k) = agent(i).new_cut_A_unbounded(1, (j-1)*env_parameters.NR_OBFLOC+k); 
                        end
                    end
                    new_cut_A_ = [new_cut_A_; new_cut_A__];
                    new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_unbounded];    
                else
                    if agent(i).isupdated == 1
                        if size(agent(i).node_boundary, 1) * size(agent(i).node_boundary, 2) > 0
                            for j = 1:1:size(agent(i).node_boundary, 2)
                                node_j = find(masteragent.node_boundary == agent(i).node_boundary(j)); 
                                for k = 1:1:env_parameters.NR_OBFLOC
                                    new_cut_A__(1, (node_j-1)*env_parameters.NR_OBFLOC+k) = agent(i).new_cut_A_bounded(1, (j-1)*env_parameters.NR_OBFLOC+k); 
                                end
                            end
                            new_cut_A__(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+i) = -1; 
                            new_cut_A_ = [new_cut_A_; new_cut_A__];
                            new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_bounded];  
                        else
                            new_cut_A__(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+i) = -1; 
                            new_cut_A_ = [new_cut_A_; new_cut_A__];
                            new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_bounded]; 
                        end
                    end
               end   
    end
        
            masteragent.cuts_A = [masteragent.cuts_A; new_cut_A_]; 
            masteragent.cuts_b = [masteragent.cuts_b; new_cut_b_]; 
end