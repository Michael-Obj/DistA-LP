function [aligned_A1, row_perms, col_perms] = align_square_matrices(A1_all, max_iterations, tolerance)
%ALIGN_SQUARE_MATRICES  Iteratively align a set of square matrices by
% row/column permutations to minimize total squared deviation.
%
%   [aligned_A1, row_perms, col_perms] = ALIGN_SQUARE_MATRICES(A1_all)
%   aligns the cell array of U matrices A1_all{1..U}, each n×n, returning
%   the aligned versions and the per-matrix row_perms{k} and col_perms{k}.
%
%   […, max_iterations]  overrides the maximum outer iterations (default 10).
%   […, …, tolerance]    sets convergence tolerance on the change in the
%                        mean matrix M (default 1e-6).
%
%   Uses the Hungarian assignment (matchpairs) to align each matrix to
%   the current mean M in turn, alternating row and column matching.

    %---- Handle inputs & defaults ----
    U = numel(A1_all);
    n = size(A1_all{1},1);
    assert(all(cellfun(@(A) isequal(size(A),[n n]), A1_all)), ...
        'All input matrices must be square of the same size.');

    if nargin < 2 || isempty(max_iterations)
        max_iterations = 10;
    end
    if nargin < 3 || isempty(tolerance)
        tolerance = 1e-6;
    end

    %---- Initialize permutations ----
    row_perms = repmat({1:n}, U, 1);
    col_perms = repmat({1:n}, U, 1);

    %---- Initial mean ----
    M = zeros(n,n);
    for k = 1:U
        M = M + A1_all{k};
    end
    M = M / U;
    prev_M = M;

    %---- Outer loop ----
    for iter = 1:max_iterations
        fprintf('Outer iteration %d\n', iter);

        % Align each matrix to M
        for k = 1:U
            rp = row_perms{k};
            cp = col_perms{k};
            A_perm = A1_all{k}(rp, cp);

            prev_cost = inf;
            % inner refinement: alternate row/col until no improvement
            for inner = 1:10
                %--- row assignment cost matrix ---
                C_row = zeros(n,n);
                for i = 1:n
                    for j = 1:n
                        C_row(i,j) = sum((A_perm(i,:) - M(j,:)).^2);
                    end
                end
                % use a large finite cost instead of Inf
                costUnmatched = 1e10;
                assign_row = matchpairs(C_row, costUnmatched, 'min');
                assign_row = sortrows(assign_row,2);
                new_rp_idx = assign_row(:,1);
                A_rows = A_perm(new_rp_idx, :);

                %--- column assignment cost matrix ---
                C_col = zeros(n,n);
                for i = 1:n
                    for j = 1:n
                        C_col(i,j) = sum((A_rows(:,i) - M(:,j)).^2);
                    end
                end
                assign_col = matchpairs(C_col, costUnmatched, 'min');
                assign_col = sortrows(assign_col,2);
                new_cp_idx = assign_col(:,1);
                A_new = A_rows(:, new_cp_idx);

                % compute cost and break if no improvement
                cost = norm(A_new - M, 'fro')^2;
                if cost >= prev_cost - tolerance
                    break;
                end
                % accept
                rp = rp(new_rp_idx);
                cp = cp(new_cp_idx);
                A_perm = A_new;
                prev_cost = cost;
            end

            row_perms{k} = rp;
            col_perms{k} = cp;
        end

        %---- update mean ----
        M_new = zeros(n,n);
        for k = 1:U
            M_new = M_new + A1_all{k}(row_perms{k}, col_perms{k});
        end
        M_new = M_new / U;

        delta = norm(M_new - prev_M, 'fro');
        fprintf('  ΔM = %g\n', delta);
        if delta < tolerance
            break;
        end
        prev_M = M_new;
        M = M_new;
    end

    %---- build final aligned outputs ----
    aligned_A1 = cell(size(A1_all));
    for k = 1:U
        aligned_A1{k} = A1_all{k}(row_perms{k}, col_perms{k});
    end
end
