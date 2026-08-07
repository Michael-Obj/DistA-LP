function [masteragent, agent, lowerbound, upperbound, upperbound_, loss, obf_matrix] = bendersDecomposition(masteragent, agent, env_parameters, ITER)


    upperbound_ = []; 
    iter = 1; 
    while iter <= ITER       
        %% Master agent calculates the boundary obfuscation vectors
        obf_matrix = zeros(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC);                                    
        if iter == 0                                                        
            obf_matrix = ones(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC)/env_parameters.NR_OBFLOC;      
            lowerbound(iter) = 0; 
        else
            %[masteragent, agent, lowerbound(iter), is_cut, loss_boundary_rec] = masterProblem(masteragent, agent, env_parameters); 
            [masteragent, agent, lowerbound(iter), is_cut] = masterProblem(masteragent, agent, env_parameters);
            obf_matrix(masteragent.node, :) = masteragent.decision;
        end        

        %% Each agent calculates the internal obfuscation vectors
        loss_internal_rec=0;

        for i = 1:1:env_parameters.NR_AGENT       
            if size(agent(i).node_internal, 2)*size(agent(i).node_internal, 1) > 0
                agent(i) = subProblem(agent(i), env_parameters, obf_matrix);
                upperbound_(iter, i) = agent(i).upperbound; 
                isunbounded(i) = agent(i).isunbounded; 
            end
            loss_internal_rec=loss_internal_rec+agent(i).loss_internal_rec_i;
        end
        upperbound(iter) = sum(upperbound_(iter, :)); 
        obf_matrix = integrateZ(agent, env_parameters, obf_matrix); 
    
        %% Each agent suggests the new cuts to the master agent
        masteragent = addNewCuts(masteragent, agent, env_parameters); 
        % for i = 1:1:env_parameters.NR_AGENT 
        %     new_cut_A__ = sparse(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+env_parameters.NR_AGENT);
        %     if agent(i).isunbounded == 1 && size(agent(i).node_boundary, 1) > 0 
        %         for j = 1:1:size(agent(i).node_boundary, 2)
        %             node_j = find(masteragent.node_boundary == agent(i).node_boundary(j)); 
        %             for k = 1:1:env_parameters.NR_OBFLOC
        %                 new_cut_A__(1, (node_j-1)*env_parameters.NR_OBFLOC+k) = agent(i).new_cut_A_unbounded(1, (j-1)*env_parameters.NR_OBFLOC+k); 
        %             end
        %         end
        %         new_cut_A_ = [new_cut_A_; new_cut_A__];
        %         new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_unbounded];    
        %     else
        %         if agent(i).isupdated == 1
        %             if size(agent(i).node_boundary, 1) * size(agent(i).node_boundary, 2) > 0
        %                 for j = 1:1:size(agent(i).node_boundary, 2)
        %                     node_j = find(masteragent.node_boundary == agent(i).node_boundary(j)); 
        %                     for k = 1:1:env_parameters.NR_OBFLOC
        %                         new_cut_A__(1, (node_j-1)*env_parameters.NR_OBFLOC+k) = agent(i).new_cut_A_bounded(1, (j-1)*env_parameters.NR_OBFLOC+k); 
        %                     end
        %                 end
        %                 new_cut_A__(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+i) = -1; 
        %                 new_cut_A_ = [new_cut_A_; new_cut_A__];
        %                 new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_bounded];  
        %             else
        %                 new_cut_A__(1, masteragent.NR_NODE_BOUNDARY*env_parameters.NR_OBFLOC+i) = -1; 
        %                 new_cut_A_ = [new_cut_A_; new_cut_A__];
        %                 new_cut_b_ = [new_cut_b_; agent(i).new_cut_b_bounded]; 
        %             end
        %         end
        %    end   
        % end
    
        % masteragent.cuts_A = [masteragent.cuts_A; new_cut_A_]; 
        % masteragent.cuts_b = [masteragent.cuts_b; new_cut_b_]; 
        if upperbound(iter) - lowerbound(iter) < 0.01
            iter;
            break; 
        end
        if (upperbound(iter) - lowerbound(iter))/lowerbound(iter) < 0.1
            break; 
        end
        iter

        iter = iter+1; 
    end
    % loss = loss_boundary_rec + loss_internal_rec;
    loss=0;
end