    % Parameters
    U = 3;   % number of users
    n = 5;   % size of location set X
    m = 8;   % size of location set Y
    lambda2 = 1.0;
    lambda3 = 1.0;

    % Generate synthetic matrices for each user
    A1_all = cell(1, U);  % distance within X
    A2_all = cell(1, U);  % distance from X to Y
    A3_all = cell(1, U);  % travel cost from X to Y

    for u = 1:U
        X = rand(n, 2) * 100;  % user-specific locations in X
        Y = rand(m, 2) * 100;  % user-specific locations in Y
        A1 = squareform(pdist(X, 'euclidean'));  % symmetric
        A2 = pdist2(X, Y, 'euclidean');          % distance
        A3 = A2 + 5 * randn(n, m);               % cost = dist + noise
        A3 = max(A3, 0);                         % non-negative

        A1_all{u} = A1;
        A2_all{u} = A2;
        A3_all{u} = A3;
    end

    % Align user matrices
    [aligned_A1, aligned_A2, aligned_A3, best_pi] = ...
        align_user_matrices(A1_all, A2_all, A3_all, lambda2, lambda3);

    fprintf('Best shared permutation: [%s]\n', num2str(best_pi));

    % Visualize original vs aligned for first user
    figure('Name', 'User 1 Alignment Visualization', 'Position', [100, 100, 1200, 600]);

    subplot(2, 3, 1); imagesc(A1_all{1}); title('Original A1^{(1)}'); colorbar;
    subplot(2, 3, 2); imagesc(A2_all{1}); title('Original A2^{(1)}'); colorbar;
    subplot(2, 3, 3); imagesc(A3_all{1}); title('Original A3^{(1)}'); colorbar;

    subplot(2, 3, 4); imagesc(aligned_A1{1}); title('Aligned A1^{(1)}'); colorbar;
    subplot(2, 3, 5); imagesc(aligned_A2{1}); title('Aligned A2^{(1)}'); colorbar;
    subplot(2, 3, 6); imagesc(aligned_A3{1}); title('Aligned A3^{(1)}'); colorbar;
