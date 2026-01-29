function is_full = is_full_rank(subproblem, constraint_matrix1, NR_LOC)
    nr_ExtremeP = size(constraint_matrix1, 1);
    constraint_matrix2 = sparse(NR_LOC, nr_ExtremeP);
    idx = 1; 
    for l = 1:1:NR_LOC
        constraint_matrix2(l, idx:idx+size(subproblem(l).ExtremeP, 2)-1) = ones(1, size(subproblem(l).ExtremeP, 2));
        idx = idx + size(subproblem(l).ExtremeP, 2); 
    end

    constraint_matrix = [constraint_matrix1; constraint_matrix2]; 
    if rank(full(constraint_matrix)) == NR_LOC*2
        is_full = 1;
    else
        is_full = 0; 
    end
end