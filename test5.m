addpath('./functions/haversine');                                           % Read the Haversine distance package. This package is created by Created by Josiah Renfree, May 27, 2010
addpath('./functions');                                                     % Read the Haversine distance package. This package is created by Created by Josiah Renfree, May 27, 2010


% Parameters
n = 10;  % Number of locations in X
m = 10; % Number of locations in Y

% Random 2D coordinates for X and Y
X = rand(n, 2) * 100; % e.g., [x, y] in [0, 100] x [0, 100]
Y = rand(m, 2) * 100;

% A1: Distance matrix between locations in X
A1 = squareform(pdist(X, 'euclidean'));

% A2: Distance matrix from X to Y
A2 = pdist2(X, Y, 'euclidean');

% A3: Travel cost from X to Y (distance + small noise or penalty)
A3 = A2 + 5*randn(n, m); % you can replace 5 with a domain-specific multiplier

% Ensure A3 is non-negative (optional)
A3 = max(A3, 0);

% Call your reordering and visualization function
[best_pi, best_params, F1, F2, F3] = reorder_fit_gaussians(A1, A2, A3, 1.0, 1.0);