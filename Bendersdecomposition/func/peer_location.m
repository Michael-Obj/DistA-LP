function peerMatrix = peer_location(G, coordinate, PATIENT, NR_LOC, ETA)
    %% Peer location search
        
    % Step 2: build the shortest path tree
    [~, D] = shortestpathtree(G, PATIENT);
    
    % % draw the shortest path tree
    % tree_edges = table2array(TR.Edges); 
    % 
    % for i = 1:1:size(tree_edges, 1)
    %     plot3(  [x_coordinate(tree_edges(i, 1)), x_coordinate(tree_edges(i, 2))], ...
    %             [y_coordinate(tree_edges(i, 1)), y_coordinate(tree_edges(i, 2))], ... 
    %             [z_coordinate(tree_edges(i, 1)), z_coordinate(tree_edges(i, 2))], ... 
    %             'k'); 
    %     hold on; 
    % end
    % plot3(x_coordinate(PATIENT), y_coordinate(PATIENT), z_coordinate(PATIENT), 'o');
    
    % Step 3: Sort the locations according to their distance to the patient's
    % location
    [D_sort, index_sort] = sort(D(1:NR_LOC)); 
    
    % bar(D_sort); 
    
    % Step 4: Find the peer locations
    peerMatrix = sparse(NR_LOC, NR_LOC); 
    
    first_idx = 1; 
    last_idx = 1;
    for i = 1:1:NR_LOC  
        while D_sort(1, first_idx) < D_sort(1, i) - ETA
            first_idx = first_idx + 1;
        end
    
        while D_sort(1, last_idx) <= D_sort(1, i) + ETA && last_idx < NR_LOC
            last_idx = last_idx + 1;
        end
        
        for j = first_idx:1:last_idx-1
            peerMatrix(index_sort(1, i), index_sort(1, j)) = 1; 
            peerMatrix(index_sort(1, j), index_sort(1, i)) = 1;
        end
    end
    
    
    %% Figure 
    
    % Target = 10; 
    % 
    % plot3(x_coordinate, y_coordinate, z_coordinate, 'o'); 
    % hold on; 
    % 
    % x_coordinate = [x_coordinate1; x_coordinate2]; 
    % y_coordinate = [y_coordinate1; y_coordinate2]; 
    % z_coordinate = [z_coordinate1; z_coordinate2]; 
    % 
    % 
    % for i = 1:1:size(edge_start, 1)
    %     [edge_start(i)+1, edge_end(i)+1]
    %     plot3(  [x_coordinate(edge_start(i)), x_coordinate(edge_end(i))], ...
    %             [y_coordinate(edge_start(i)), y_coordinate(edge_end(i))], ... 
    %             [z_coordinate(edge_start(i)), z_coordinate(edge_end(i))], ... 
    %             'k'); 
    %     hold on; 
    % end
    % 
    % peer_vector = find(peer_matrix(Target, :) == 1); 
    % 
    % plot3(x_coordinate(peer_vector), y_coordinate(peer_vector), z_coordinate(peer_vector), '*'); 
    % hold on; 

end
