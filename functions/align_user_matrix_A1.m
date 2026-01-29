function [aligned_A1, best_pi] = align_user_matrix_A1(A1_all)
% Aligns square matrices A1 across users by finding the best shared row/column permutation
% Input:
%   A1_all: cell array of U matrices (n x n)
% Output:
%   aligned_A1: aligned matrices after permutation (cell array)
%   best_pi: best shared permutation of row/column indices


    U = numel(A1_all);
    n = size(A1_all{1}, 1);
    perms_I = perms(1:n);
    K = size(perms_I, 1);
    best_loss = inf;
    
    for i = 1:K
        pi = perms_I(i, :);
        total_loss = 0;
    
        for u = 1:U
            A1p{u} = A1_all{u}(pi, pi);
        end
    
        A1_mean = mean(cat(3, A1p{:}), 3);
    
        for u = 1:U
            total_loss = total_loss + sum(abs(A1p{u} - A1_mean), 'all');
        end
    
        if total_loss < best_loss
            best_loss = total_loss;
            best_pi = pi;
            best_A1p = A1p;
        end
    end
    
    aligned_A1 = best_A1p;
end
