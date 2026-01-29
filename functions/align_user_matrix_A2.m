function [aligned_A2, best_pi] = align_user_matrix_A2(A2_all)
% Aligns rectangular matrices A2 across users by finding the best shared row permutation
% Input:
%   A2_all: cell array of U matrices (n x m)
% Output:
%   aligned_A2: aligned matrices after permutation (cell array)
%   best_pi: best shared permutation of row indices

    U = numel(A2_all);
    [n, m] = size(A2_all{1});
    perms_I = perms(1:n);
    K = size(perms_I, 1);
    best_loss = inf;
    
    for i = 1:K
        pi = perms_I(i, :);
        total_loss = 0;
    
        for u = 1:U
            A2p{u} = A2_all{u}(pi, :);
        end
    
        A2_mean = mean(cat(3, A2p{:}), 3);
    
        for u = 1:U
            total_loss = total_loss + sum(abs(A2p{u} - A2_mean), 'all');
        end
    
        if total_loss < best_loss
            best_loss = total_loss;
            best_pi = pi;
            best_A2p = A2p;
        end
    end
    
    aligned_A2 = best_A2p;
end