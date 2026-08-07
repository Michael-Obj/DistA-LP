function [masteragent, agent, lowerbound, upperbound] = bendersDecomposition_RLTraining(env_parameters, rl_parameters, masteragent, agent)

    %% This function is used to train the RL policy
    masteragent = myReset(masteragent); 

    % ---------------------- Initialize learning agents

    for i = 1:1:env_parameters.NR_AGENT
        policy(i) = struct('probability', ones(1, size(agent(i).extremerays, 2)));
    end

    % env = myEnvironment(env_parameters, rl_parameters, agent, masteragent); 

    %% Monte-Carlo algorithm
    for ep_idx = 1:1:rl_parameters.NR_EPISODE
    % for task_loc = 1:1:1
        for i = 1:1:env_parameters.NR_AGENT                                 % Initialize the parameters of episode
            episode(ep_idx, i) = struct('states', zeros(env_parameters.ITER, size(agent(i).extremerays, 2)), ...
                                        'isdone', zeros(env_parameters.ITER, 1), ...
                                        'actions', zeros(env_parameters.ITER, 1), ...
                                        'instant_reward', zeros(env_parameters.ITER, 1), ... 
                                        'reward', zeros(env_parameters.ITER, 1)); 
            agent(i).isunbounded = 0;
        end

        masteragent = myReset(masteragent); 
        upperbound_details = []; 

        iter = 1; 
        extreme_idx = zeros(1, env_parameters.NR_AGENT); 
        while iter <= env_parameters.ITER
            [ep_idx iter] 
            obfuscation_matrix = zeros(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC); 

            %% Master program
            if iter == 0
                obfuscation_matrix = ones(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC)/env_parameters.NR_OBFLOC; 
                lowerbound(iter) = 0; %1/20*sum(sum(cost_matrix)); 
            else
                [masteragent, agent, lowerbound(iter), is_cut] = masterProblem(masteragent, agent, env_parameters); 
                obfuscation_matrix(masteragent.node, :) = masteragent.decision;
            end

            
            %% Subproblems
            for i = 1:1:env_parameters.NR_AGENT       
                if size(agent(i).node_boundary  , 2)*size(agent(i).node_boundary, 1) > 0
                    [agent(i), episode(ep_idx, i)] = subProblemRL(agent(i), policy(i), env_parameters.cost_matrix, env_parameters.NR_OBFLOC, obfuscation_matrix, episode(ep_idx, i), iter);
                    upperbound_details(iter, i) = agent(i).upperbound; 
                    isunbounded(i) = agent(i).isunbounded; 
                end
            end

            %% Each agent suggests the new cuts to the master agent
            masteragent = addNewCuts(masteragent, agent, env_parameters); 


            upperbound(iter) = sum(upperbound_details(iter, :)); 
            obfuscation_matrix = integrateZ(agent, env_parameters); 

            if upperbound(iter) - lowerbound(iter) < 0.01
                break; 
            end
            iter = iter+1; 
        end
        episode(ep_idx, :) = reward_compute(episode(ep_idx, :)); 
    end
end