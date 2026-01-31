function cost_distribution = cost_error_distribution(z, cost_matrix, NR_LOC)
    NR_SAMPLE = 100; 
    cost_distribution = zeros(NR_LOC, NR_SAMPLE); 
    for i = 1:1:NR_LOC
        z_distribution = cumsum(z(i, :));
        rand_value = rand(1, NR_SAMPLE); 
        for k = 1:1:NR_SAMPLE
            for j = 1:1:NR_LOC-1
                if rand_value(1, k) <= z_distribution(1, j+1)  && rand_value(1, k) > z_distribution(1, j)
                    cost_distribution(i, k) = cost_matrix(i, j); 
                end
            end
        end
    end
    cost_distribution = reshape(cost_distribution, 1, NR_LOC*NR_SAMPLE); 
end