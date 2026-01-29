addpath('./functions/');

A = rand(10, 15); % example input matrix
[G_fit, params, pi_opt] = fit_gaussian_lower_bound(A, 1000);

imagesc(G_fit); colorbar;
title('Gaussian Approximation (Lower Bound)');
