function extremepoint = add_extremepoint_rand(GeoIMatrix, peerMatrix, nr_constraints, iter, k, NR_LOC)
    b = zeros(nr_constraints, 1); 
    lb = zeros(NR_LOC, 1)*0;
    ub = full(peerMatrix(:, k)); 


    options = optimoptions('linprog','Algorithm','dual-simplex', 'Display','none');
    options.Preprocess = 'none';
    if k >= 1 
        for i = 1:1:1
            f = zeros(1, NR_LOC); 
            f(1, iter) = -1;
            [extremepoint, ~] = linprog(f, GeoIMatrix, b, [], [], lb, ub, options); 
%            extremepoint = [extremepoint, extremepoint_temp]; 
        end
    end
end