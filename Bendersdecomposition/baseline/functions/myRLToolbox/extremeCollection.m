function [masteragent, agent, lowerbound, upperbound, nr_extremepoints] = extremeCollection(env_parameters, masteragent, agent, ITER)
                                                                            %masteragent, agent, node_boundary, ...
                                                                            % node_in_target, obf_loc, ...
                                                                           
%% This function is used to collect the extreme points of all the agents
%     load('.\Dataset\rome\intermediate\agent.mat'); 
    load('.\Dataset\rome\intermediate\obf_loc.mat'); 
    load('.\Dataset\rome\intermediate\node_in_target.mat'); 
%    load('.\Dataset\rome\intermediate\masteragent.mat'); 
%    load('.\Dataset\rome\intermediate\node_boundary.mat'); 
    load('.\Dataset\rome\intermediate\G.mat'); 

    nr_extremepoints = zeros(env_parameters.NR_AGENT, 100);
%     load('.\Dataset\rome\intermediate\agent_extreme.mat');
%     load('.\Dataset\rome\intermediate\nr_extremepoints.mat');
%     load('.\Dataset\rome\intermediate\masteragent_extreme.mat');

     
    for task_loc = 1:1:1
        % clear cost_matrix; 
        masteragent.cuts_A = [];
        masteragent.cuts_b = [];
        masteragent.decision = [];
        % cost_matrix = costMatrix(node_in_target, task_loc, obf_loc, G); 
        upperbound_ = []; 
        iter = 1; 
        while iter <= ITER
            [task_loc iter]
            Z = zeros(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC); 
            % Master program calculates the inter set decision variables
            if iter == 0
                Z = ones(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC)/env_parameters.NR_OBFLOC; 
                lowerbound(iter) = 0; %1/20*sum(sum(cost_matrix)); 
            else
                [masteragent, agent, lowerbound(iter), is_cut] = masterProblem(masteragent, agent, env_parameters); 
                Z(masteragent.node, :) = masteragent.decision;
            end
            new_cut_A_ = []; 
            new_cut_b_ = []; 
            
        
            for i = 1:1:env_parameters.NR_AGENT       
                if size(agent(i).node_boundary, 2)*size(agent(i).node_boundary, 1) > 0
                    agent(i) = subProblemExtremeCollection(agent(i), env_parameters, Z);
                    upperbound_(iter, i) = agent(i).upperbound; 
                    isunbounded(i) = agent(i).isunbounded; 
                end
            end
            upperbound(iter) = sum(upperbound_(iter, :)); 
    
            Z = integrateZ(agent, env_parameters); 
        
        
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
            % if upperbound(iter) - lowerbound(iter) < 0.01
            %     break; 
            % end
            iter = iter+1; 
        end
        for i = 1:1:env_parameters.NR_AGENT
            nr_extremepoints(i, task_loc) = size(agent(i).extremerays,2); 
        end
    end
end