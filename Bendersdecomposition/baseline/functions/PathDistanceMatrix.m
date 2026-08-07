function path_distance_matrix = PathDistanceMatrix(node_in_target, G)
    NR_NODE_IN_TARGET = size(node_in_target, 2); 
    path_distance_matrix = -ones(NR_NODE_IN_TARGET, NR_NODE_IN_TARGET);  
    for i = 1:1:NR_NODE_IN_TARGET
        for j = 1:1:NR_NODE_IN_TARGET
            [i j]
            [~, path_distance_matrix(i, j)] = shortestpath(G, node_in_target(1, i), node_in_target(1, j)); 
        end
    end
end