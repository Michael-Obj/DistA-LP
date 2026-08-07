function [agent, adjacence_matrix_intraset, adjacence_matrix_interset] ...
    = agentCreation(cluster_idx, node_in_target, adjacence_matrix, distance_matrix, NR_AGENT, NR_NODE_IN_TARGET, NR_OBFLOC, EPSILON, beta, xi_hathat)
    for i = 1:1:NR_AGENT
        agent(i) = struct(  'idx', i, ...
                            'node', [], ...
                            'node_intraset', [], ...
                            'node_internal', [], ... %replace node_intraset in the end
                            'node_interset', [], ...
                            'node_boundary', [], ... %replace node_interset in the end
                            'cost_vector', [], ...
                            'intraset', [], ...
                            'internal', [], ...  % replace the intraset
                            'interset', [], ...
                            'boundary', [], ...  % replace the internal
                            'new_cut_A_bounded', [], ...
                            'new_cut_b_bounded', 0, ...
                            'new_cut_A_unbounded', [], ...
                            'new_cut_b_unbounded', 0, ...
                            'GeoI', [], ...
                            'decision', [], ...
                            'isunbounded', 0, ...
                            'isupdated', 0, ...
                            'upperbound', 9999, ...
                            'extremerays', [], ... 
                            'loss_internal_rec_i', 0, ...
                            'z', 0); 
    end
    for i = 1:1:size(node_in_target, 2)
        agent(cluster_idx(i, 1)).node = [agent(cluster_idx(i, 1)).node, i];
    end
    
    adjacence_matrix_intraset = zeros(NR_NODE_IN_TARGET, NR_NODE_IN_TARGET); 
    adjacence_matrix_interset = zeros(NR_NODE_IN_TARGET, NR_NODE_IN_TARGET); 
                                                                                % Determine the inter-set nodes and intra-set nodes for each agent
    [idx_i, idx_j, ~] = find(adjacence_matrix);  
    for l = 1:1:size(idx_i, 1) 
        if cluster_idx(idx_i(l, 1)) == cluster_idx(idx_j(l, 1))
                                                                                % Find the intra set nodes for each agent
            agent(cluster_idx(idx_i(l, 1))).intraset = [agent(cluster_idx(idx_i(l, 1))).intraset; [idx_i(l, 1), idx_j(l, 1)]];
            adjacence_matrix_intraset(idx_i(l, 1), idx_j(l, 1)) = 1; 
        else
                                                                                % Find the inter set nodes for each agent
            agent(cluster_idx(idx_i(l, 1))).interset = [agent(cluster_idx(idx_i(l, 1))).interset; [idx_i(l, 1), idx_j(l, 1)]];
            adjacence_matrix_interset(idx_i(l, 1), idx_j(l, 1)) = 1; 
            if size(find(agent(cluster_idx(idx_i(l, 1))).node_interset == idx_j(l, 1)), 2) == 0
                agent(cluster_idx(idx_i(l, 1))).node_interset = [agent(cluster_idx(idx_i(l, 1))).node_interset, idx_i(l, 1)];
            end
        end
    end
    for i = 1:1:NR_AGENT
        agent(i).node = unique(agent(i).node); 
        agent(i).node_interset = unique(agent(i).node_interset); 
        agent(i).node_intraset = unique(setdiff(agent(i).node, agent(i).node_interset));
    end
    for i = 1:1:NR_AGENT
%    i
        agent(i) = creatSubGeoI(agent(i), distance_matrix, NR_OBFLOC, EPSILON, beta, xi_hathat); 
    end
    for i = 1:1:NR_AGENT
        agent(i).node_internal=agent(i).node_intraset;
        agent(i).node_boundary=agent(i).node_interset;
        agent(i).internal=agent(i).intraset;
        agent(i).boundary=agent(i).interset;
        % agent(i) = rmfield(agent(i), 'node_intraset');
        % agent(i) = rmfield(agent(i), 'node_interset');
        % agent(i) = rmfield(agent(i), 'intraset');
        % agent(i) = rmfield(agent(i), 'interset');
    end
end