% Example matrices
A1 = rand(5, 12);
A2 = A1(randperm(5), :) + 0.1 * randn(5, 12);
A3 = A1(randperm(5), :) + 0.2 * randn(5, 12);

matrices = {A1, A2, A3};
max_iters = 100;

[aligned_mats, perms] = align_matrices_rowwise(matrices, max_iters);
