function cost_matrix = costMatrix_coarse(target_min, task_loc, obf_loc, G, all_target, node_in_target)
    NR_LOC = length(all_target); 
    NR_OBFLOC = size(obf_loc, 2); 
    cost_matrix = zeros(NR_LOC, NR_OBFLOC); 
    for i = 1:1:NR_LOC
        [~, D] = shortestpathtree(G, node_in_target(task_loc)); % road
        %[~, D] = shortestpathtree(G, 1); % grid
        for j = 1:1:NR_OBFLOC         
            %%% [~,c_i] = shortestpathtree(G, node_in_target(task_loc), node_in_target(i)); 
            %%% [~,c_j] = shortestpathtree(G, node_in_target(task_loc), node_in_target(obf_loc(j))); 
            cost_matrix(i,j) = abs(D(node_in_target(target_min(i)))-D(node_in_target(obf_loc(j)))); 
            %cost_matrix(i,j) = 10000*abs(D(i)-D(j));
        end
    end
    cost_matrix = cost_matrix/NR_LOC; 
end