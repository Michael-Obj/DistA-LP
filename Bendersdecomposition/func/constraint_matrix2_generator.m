function constraint_matrix2 = constraint_matrix2_generator(subproblem, nr_ExtremeP, NR_LOC)
    constraint_matrix2 = sparse(NR_LOC, nr_ExtremeP); 
    idx = 1; 
    for l = 1:1:NR_LOC
        constraint_matrix2(l, idx:idx+size(subproblem(l).ExtremeP, 2)-1) = ones(1, size(subproblem(l).ExtremeP, 2));
        idx = idx + size(subproblem(l).ExtremeP, 2); 
    end
end