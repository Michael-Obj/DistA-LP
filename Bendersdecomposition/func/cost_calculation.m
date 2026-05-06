function costVector = cost_calculation(G, PATIENT, NR_LOC)
    costVector = zeros(NR_LOC*NR_LOC, 1); 
    [~, D] = shortestpathtree(G, PATIENT);
    for i = 1:1:NR_LOC
        for j = 1:1:NR_LOC
            costVector((i-1)*NR_LOC + j, 1) = abs(D(1, i) - D(1, j)); 
        end
    end
end