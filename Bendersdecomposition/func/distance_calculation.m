function distMatrix = distance_calculation(G, peerMatrix, NR_LOC)
    distMatrix = sparse(NR_LOC, NR_LOC); 
    
    [row_idx, col_idx] = find(peerMatrix); 

    for i = 1:1:size(row_idx, 1)
        [~, distMatrix(row_idx(i), col_idx(i))] = shortestpath(G, row_idx(i), col_idx(i)); 
    end
end