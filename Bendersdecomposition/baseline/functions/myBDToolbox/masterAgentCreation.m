function masteragent = masterAgentCreation(distance_matrix, agent, adjacence_matrix, cluster_idx, NR_NODE_IN_TARGET, NR_OBFLOC, NR_AGENT, EPSILON, beta, xi_hathat)
   
    adjacence_matrix_intraset = zeros(NR_NODE_IN_TARGET, NR_NODE_IN_TARGET); 
    adjacence_matrix_boundary = zeros(NR_NODE_IN_TARGET, NR_NODE_IN_TARGET); 
    [idx_i, idx_j, ~] = find(adjacence_matrix);  
    for l = 1:1:size(idx_i, 1) 
        if cluster_idx(idx_i(l, 1)) == cluster_idx(idx_j(l, 1))
            adjacence_matrix_intraset(idx_i(l, 1), idx_j(l, 1)) = 1; 
        else
            adjacence_matrix_boundary(idx_i(l, 1), idx_j(l, 1)) = 1; 
        end
    end
    
    % Find the whole set of inter-set nodes
    node_boundary = find(sum(adjacence_matrix_boundary)/max(sum(adjacence_matrix_boundary))>0);
    NR_NODE_INTERSET = size(node_boundary, 2);
    % Find the whole set of intra-set nodes
    node_intraset = find(sum(adjacence_matrix_boundary)/max(sum(adjacence_matrix_boundary))==0);
    NR_NODE_INTRASET = size(node_intraset, 2);

    masteragent = struct(  'node', node_boundary, ...
                           'cost_vector', [], ...
                           'adjacence_matrix_boundary', [], ...
                           'GeoI', [], ...
                           'decision', [], ...
                           'cuts_A', [], ...
                           'cuts_b', [], ...
                           'node_boundary', node_boundary, ...
                           'NR_NODE_INTERSET', NR_NODE_INTERSET, ...
                           'NR_NODE_BOUNDARY', NR_NODE_INTERSET);
    masteragent.adjacence_matrix_boundary = adjacence_matrix_boundary;
    masteragent = creatMasterGeoI(masteragent, distance_matrix, NR_OBFLOC, EPSILON, beta, xi_hathat);
    
    % for l = 1:1:NR_AGENT
    %     % if size(agent(l).node_boundary, 1) >0 && size(agent(l).node_intraset, 1) >0
    %         masteragent = creatMasterGeoI_improve(masteragent, agent(l), adjacence_matrix, distance_matrix, NR_OBFLOC, NR_AGENT, EPSILON); 
    %     end
    % end
end