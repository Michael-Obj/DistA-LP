function [GeoIMatrix, extremepoint] = DW_init(peerMatrix, peerNeighbor, distMatrix, k, NR_LOC, EPSILON, iter)
    extremepoint = []; 
    [idx_row, idx_col] = find(peerMatrix); 
    real_loc_idx = idx_row(find(idx_col == k));

    GeoIMatrix = sparse(0, NR_LOC); 
    constraint_idx = 1; 
    for i = 1:1:size(real_loc_idx, 1)-1
        for j = i+1:1:size(real_loc_idx, 1)
            % [real_loc_idx(i) real_loc_idx(j)]
            if peerNeighbor(real_loc_idx(i), real_loc_idx(j)) > 0
                GeoIMatrix(constraint_idx, real_loc_idx(i)) = 1;
                GeoIMatrix(constraint_idx, real_loc_idx(j)) = -exp(EPSILON*distMatrix(real_loc_idx(i), real_loc_idx(j)));
                constraint_idx = constraint_idx + 1;
                GeoIMatrix(constraint_idx, real_loc_idx(j)) = 1;
                GeoIMatrix(constraint_idx, real_loc_idx(i)) = -exp(EPSILON*distMatrix(real_loc_idx(i), real_loc_idx(j)));
                constraint_idx = constraint_idx + 1;
            end
        end
    end
    nr_constraints = constraint_idx - 1;
    for iter = 1:1:NR_LOC
        extremepoint_temp = add_extremepoint_rand(GeoIMatrix, peerMatrix, nr_constraints, iter, k, NR_LOC); 
        extremepoint = [extremepoint, extremepoint_temp]; 
    end
end