% visualize_permuted_and_fitted_distance_matrix.m
% This script generates random 2D points, computes their pairwise distance matrix,
% permutes the distance matrix without fitting, and then fits a 2D Gaussian to the distance matrix.

addpath('./functions/');

% Step 1: Generate Random 2D Locations
N = 20; % Number of points
xRange = [0, 100];
yRange = [0, 100];

% Generate random (x, y) coordinates
x = xRange(1) + (xRange(2) - xRange(1)) * rand(N, 1);
y = yRange(1) + (yRange(2) - yRange(1)) * rand(N, 1);

% Combine into a single matrix of coordinates
locations = [x, y];

% Step 2: Compute the Distance Matrix
% Compute pairwise Euclidean distance matrix
D = pdist2(locations, locations)/10;

% Generate utiltiy loss matrix
UL = rand(N, 20); 

% % Step 3: Permute the Distance Matrix Without Fitting
% % Generate a random permutation of indices
% perm_indices = randperm(N);
% 
% % Apply the permutation to both rows and columns to maintain symmetry
% D_permuted = D(perm_indices, perm_indices);

% Step 4: Fit a 2D Gaussian to the Original Distance Matrix
max_iters = 2000; % Number of iterations for the fitting algorithm

% Fit the Gaussian to the original distance matrix
[G_fit, params, pi_opt] = fit_gaussian_to_matrix(D, max_iters);

% Permute the original distance matrix according to the optimal permutation
D_fitted_permuted = D(pi_opt, pi_opt);


[P_real, expected_loss_real] = data_perturbation(UL(pi_opt, :), D_fitted_permuted, 1); 

[P_fit, expected_loss_fit] = data_perturbation(UL(pi_opt, :), G_fit, 1); 


% Visualization
figure;

subplot(1, 3, 1);
imagesc(D);
colorbar;
title('Randomly Permuted Distance Matrix');

subplot(1, 3, 2);
imagesc(D_fitted_permuted);
colorbar;
title('Fitted Permuted Distance Matrix');

subplot(1, 3, 3);
imagesc(G_fit);
colorbar;
title('Fitted Gaussian Matrix');
