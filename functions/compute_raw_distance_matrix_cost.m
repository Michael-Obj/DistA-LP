function raw_distance_matrix = compute_raw_distance_matrix_cost(longitude, latitude, target_long, target_lat, df_edges, df_nodes)
    num_locations = length(longitude);

    % Create the graph
    G = create_graph_from_edges(df_edges, df_nodes);
    
    % Calculate shortest path distances between locations and target
    raw_distance_matrix = zeros(num_locations, 1);
    for i = 1:num_locations
        start_node = find_nearest_node(longitude(i), latitude(i), df_nodes);
        target_node = find_nearest_node(target_long, target_lat, df_nodes);

        % Use MATLAB's built-in shortest path function
        [~, raw_distance_matrix(i)] = shortestpath(G, start_node, target_node);  
        raw_distance_matrix(i) = raw_distance_matrix(i)/1000; 
    end
end
