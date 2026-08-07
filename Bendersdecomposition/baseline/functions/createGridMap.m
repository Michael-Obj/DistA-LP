function [adjacence_matrix, distance_matrix, coord, distance_graph, env_parameters] = createGridMap(env_parameters)
    %% coord: all points
    % cell_size = 1.5; 
    % for i = 1:1:30
    %     for j = 1:1:50
    %         idx = 50*(i - 1) + j;
    %         coord(idx, 1) = i*cell_size;
    %         coord(idx, 2) = j*cell_size;
    %     end
    % end
    %%
    opts = detectImportOptions('.\Dataset\grid_map\london_cell_node.csv');
    opts = setvaropts(opts, {'y', 'x'}, 'Type', 'double'); 
    tbl = readtable('.\Dataset\grid_map\london_cell_node.csv', opts);
    grid_index = tbl.grid_index;
    coord(:, 1) = tbl.x;  
    coord(:, 2) = tbl.y;  
    %%
    distance_matrix = zeros(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_NODE_IN_TARGET); 
    for i = 1:1:env_parameters.NR_NODE_IN_TARGET
        for j = 1:1:env_parameters.NR_NODE_IN_TARGET
            distance_matrix(i,j) = sqrt(sum((coord(i, :) - coord(j, :)).^2)); 
        end
    end
    % load('.\Dataset\rome\intermediate\obf_loc.mat'); 
   

    % env_parameters.obf_loc = obf_loc; 
    adjacence_matrix = heaviside(1 - distance_matrix/env_parameters.NEIGHBOR_THRESHOLD);       % Create the adjacency matrix. 

    distance_matrix_= distance_matrix.*adjacence_matrix;
    G = graph(distance_matrix_); 


    for i = 1:1:env_parameters.NR_NODE_IN_TARGET
        [~, D] = shortestpathtree(G, 1); 
        for j = 1:1:env_parameters.NR_OBFLOC         
            % [~,c_i] = shortestpathtree(G, node_in_target(task_loc), node_in_target(i)); 
            % [~,c_j] = shortestpathtree(G, node_in_target(task_loc), node_in_target(obf_loc(j))); 
            cost_matrix(i,j) = 10000*abs(D(i)-D(j)); 
        end
    end
    cost_matrix = cost_matrix/env_parameters.NR_OBFLOC; 
    env_parameters.cost_matrix = cost_matrix; %abs(randn(env_parameters.NR_NODE_IN_TARGET, env_parameters.NR_OBFLOC));             % Calculate the cost matrix 
    %% cost
    env_parameters.cost_matrix = distance_matrix;
    %%
    distance_graph = distances(G); 
end