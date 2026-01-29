% Generate test matrices with known base

addpath('./functions/');

m = 5; n = 4; K = 10;
base = rand(m, n);
matrices = cell(1, K);
original_row_perms = cell(1, K);
original_col_perms = cell(1, K);

for k = 1:K
    % Create random permutations
    [~, original_row_perms{k}] = sort(rand(1, m));
    [~, original_col_perms{k}] = sort(rand(1, n));
    
    % Create permuted matrix with noise
    matrices{k} = base(original_row_perms{k}, original_col_perms{k}) + 0.1*randn(m, n);
end

% Compute initial differences
fprintf('=== Before Alignment ===\n');
disp('Pairwise differences between original matrices:');
original_pairwise_diffs = zeros(K, K);
for i = 1:K
    for j = 1:K
        original_pairwise_diffs(i,j) = norm(matrices{i} - matrices{j}, 'fro');
    end
end
disp(original_pairwise_diffs);

% Compute differences from base (for demonstration)
fprintf('\nDifferences from base matrix:\n');
original_base_diffs = zeros(1, K);
for k = 1:K
    original_base_diffs(k) = norm(matrices{k} - base, 'fro');
end
disp(original_base_diffs);

% Align matrices
[aligned, row_perms, col_perms] = align_matrices(matrices, 100, 1e-10);

% Compute post-alignment differences
fprintf('\n=== After Alignment ===\n');
disp('Pairwise differences between aligned matrices:');
aligned_pairwise_diffs = zeros(K, K);
for i = 1:K
    for j = 1:K
        aligned_pairwise_diffs(i,j) = norm(aligned{i} - aligned{j}, 'fro');
    end
end
disp(aligned_pairwise_diffs);

% Compute new differences from base
fprintf('\nDifferences from base matrix after alignment:\n');
aligned_base_diffs = zeros(1, K);
for k = 1:K
    aligned_base_diffs(k) = norm(aligned{k} - base, 'fro');
end
disp(aligned_base_diffs);

% Visual comparison for first matrix
k = 1;  % Example matrix index
fprintf('\n=== Visual Comparison for Matrix %d ===\n', k);
disp('Original shuffled matrix:');
disp(matrices{k});
disp('Aligned matrix:');
disp(aligned{k});
disp('Original base matrix:');
disp(base);

% Display permutation information
fprintf('\nRow permutation for matrix %d (original -> aligned):\n', k);
disp([original_row_perms{k}' row_perms{k}']);

fprintf('Column permutation for matrix %d (original -> aligned):\n', k);
disp([original_col_perms{k}' col_perms{k}']);