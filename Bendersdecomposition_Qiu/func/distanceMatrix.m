function distance_matrix = distanceMatrix(x, y)
    NR_LOC = size(x, 1); 
    distance_matrix = zeros(NR_LOC, NR_LOC); 
    for i = 1:1:NR_LOC
        i
        for j = i+1:1:NR_LOC
            if x
            [distance, ~, ~] = haversine([x(i,1), y(i,1)], [x(j,1), y(j,1)]); 
            % if distance <= NEIGHBOR_THRESHOLD
                distance_matrix(i, j) = distance;
                distance_matrix(j, i) = distance;
            % end
        end
    end
end