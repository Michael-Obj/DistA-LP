function indist_set = threatByCostMatrix(cr_table, CRT_GRID_CELL_SIZE, SAMPLE_SIZE)
    cr_vector = reshape(cr_table.approximated_cost, 1, size(cr_table.approximated_cost, 1)*size(cr_table.approximated_cost, 2)); 
    cr_vector_sort = sort(cr_vector); 
 
    for i = 1:1:SAMPLE_SIZE
        idx = randi(size(cr_vector_sort, 2));
        cost_est = cr_vector_sort(1, idx); 
        % cost_est =20; 
        indist_nr = 0; 
        for j = 1:1:size(cr_vector_sort, 2)
            if cr_vector_sort(1, j) <= cost_est+CRT_GRID_CELL_SIZE*1.42 && cr_vector_sort(1, j) >= cost_est-CRT_GRID_CELL_SIZE*1.4
                indist_nr = indist_nr+1;
            end
        end
        indist_set(1, i) = indist_nr; 
    end
    indist_set = indist_set/size(cr_vector_sort, 2)*size(cr_vector_sort, 2); 
end