function [aligned_matrices, row_perms, col_perms] = align_matrices(matrices, max_iterations, tolerance)
    % Convert 3D array to cell if needed, then validate input
    if ~iscell(matrices)
        [m, n, K] = size(matrices);
        matrices_cell = cell(1, K);
        for k = 1:K
            matrices_cell{k} = matrices(:,:,k);
        end
        matrices = matrices_cell;
    else
        K = numel(matrices);
        [m, n] = size(matrices{1});
        for k = 2:K  % Validate consistent dimensions
            assert(isequal(size(matrices{k}), [m, n]), 'All matrices must have the same dimensions');
        end
    end


    % Parameters
    if nargin < 2
        max_iterations = 10;
    end
    if nargin < 3
        tolerance = 1e-6;
    end


    % Initialize permutations
    row_perms = cell(K, 1);
    col_perms = cell(K, 1);
    for k = 1:K
        row_perms{k} = 1:m;
        col_perms{k} = 1:n;
    end


    % Compute initial average
    M = mean(cat(3, matrices{:}), 3);
    prev_M = M;
    converged = false;
    iteration = 0;


    while ~converged && iteration < max_iterations
        iteration = iteration + 1;
        fprintf('Iteration %d\n', iteration);


        % Align each matrix to M
        for k = 1:K
            row_perm_current = row_perms{k};
            col_perm_current = col_perms{k};
            A_perm = matrices{k}(row_perm_current, col_perm_current);
            prev_cost = inf;
            iter_inner = 0;
            max_inner_iter = 10;


            while iter_inner < max_inner_iter
                iter_inner = iter_inner + 1;


                % Align rows
                C_row = zeros(m, m);
                for i = 1:m
                    for j = 1:m
                        C_row(i, j) = sum((A_perm(i, :) - M(j, :)).^2);
                    end
                end
                assignments = matchpairs(C_row, 1e10, 'min');
                if size(assignments, 1) < m
                    error('Row assignment failed for matrix %d', k);
                end
                sorted_assignments = sortrows(assignments, 2);
                P_temp = sorted_assignments(:, 1);
                A_perm_rows = A_perm(P_temp, :);


                % Align columns
                C_col = zeros(n, n);
                for i = 1:n
                    for j = 1:n
                        C_col(i, j) = sum((A_perm_rows(:, i) - M(:, j)).^2);
                    end
                end
                assignments = matchpairs(C_col, 1e10, 'min');
                if size(assignments, 1) < n
                    error('Column assignment failed for matrix %d', k);
                end
                sorted_assignments = sortrows(assignments, 2);
                Q_temp = sorted_assignments(:, 1);
                A_perm_new = A_perm_rows(:, Q_temp);


                % Check convergence
                curr_cost = norm(A_perm_new - M, 'fro')^2;
                if curr_cost >= prev_cost - tolerance
                    break;
                end


                % Update permutations and A_perm
                row_perm_current = row_perm_current(P_temp);
                col_perm_current = col_perm_current(Q_temp);
                A_perm = A_perm_new;
                prev_cost = curr_cost;
            end


            % Save permutations
            row_perms{k} = row_perm_current;
            col_perms{k} = col_perm_current;
        end


        % Update the average matrix
        new_M = zeros(m, n);
        for k = 1:K
            new_M = new_M + matrices{k}(row_perms{k}, col_perms{k});
        end
        new_M = new_M / K;


        % Check for convergence
        delta_M = norm(new_M - prev_M, 'fro');
        fprintf('Delta M: %f\n', delta_M);
        if delta_M < tolerance
            converged = true;
        else
            prev_M = new_M;
            M = new_M;
        end
    end


    % Generate aligned matrices (cell array output)
    aligned_matrices = cell(1, K);
    for k = 1:K
        aligned_matrices{k} = matrices{k}(row_perms{k}, col_perms{k});
    end
end
