function cluster_idx = gridclustering(loc_x_in_target, loc_y_in_target, cluster_rnd_idx, X_SIZE, TARGET_X_MIN, Y_SIZE, TARGET_Y_MIN, NR_AGENT_ROW, NR_AGENT_COL)
    cluster_idx = zeros(size(loc_x_in_target)); 
    X_CELL_SIZE = X_SIZE/NR_AGENT_COL; 
    Y_CELL_SIZE = Y_SIZE/NR_AGENT_ROW;
    for i = 1:1:size(loc_x_in_target, 1)
        idx_col = ceil((loc_x_in_target(i) - TARGET_X_MIN)/X_CELL_SIZE);
        idx_row = ceil((loc_y_in_target(i) - TARGET_Y_MIN)/Y_CELL_SIZE);
        cluster_idx(i, 1) = cluster_rnd_idx((idx_row-1)*NR_AGENT_COL + idx_col); 
    end
end