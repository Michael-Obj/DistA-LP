% Generate 10 random 2D points
points = rand(10, 2) * 100;
D = squareform(pdist(points));

% Reorder for Gaussian-like structure
Y = mdscale(D, 1);
[~, order] = sort(Y);
D_reordered = D(order, order);

% Visualize with dots and Gaussian fit
plotDotsWithGaussianGrid(D_reordered);
