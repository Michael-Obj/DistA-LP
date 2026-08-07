function plotSetEdges(loc_x_in_target, loc_y_in_target, cluster_idx, adjacence_matrix_intraset)
    gplot(adjacence_matrix_intraset, [loc_x_in_target'; loc_y_in_target']'); 
    hold on; 
    scatter(loc_x_in_target, loc_y_in_target, [], cluster_idx); 
end