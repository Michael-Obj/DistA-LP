function [A1_hat, A2_hat, A3_hat] = recover_rbfs_approximations(n, m, params, num_centres, sigma, pi)
% Recover approximated A1 (n×n), A2 (n×m), A3 (n×m) in the ORIGINAL order
% from RBF weights and the (row) permutation `pi` used during fitting.
%
% Inputs
%   n, m        : sizes (A1 is n×n; A2,A3 are n×m)
%   params      : 1×(3L) vector [α₁…α_L , β₁…β_L , γ₁…γ_L]
%   num_centres : L (same as used in reorder_fit_rbfs)
%   sigma       : RBF bandwidth σ (same as used in reorder_fit_rbfs)
%   pi          : 1×n permutation applied during fitting
%
% Outputs
%   A1_hat, A2_hat, A3_hat in ORIGINAL row/column order.

    if nargin < 4 || isempty(num_centres), num_centres = 9; end
    if nargin < 5 || isempty(sigma),       sigma       = max(n)/3; end

    L = num_centres;

    if numel(params) ~= 3*L
        error('recover_rbfs_approximations:badParamLen', ...
              'params must have length %d (got %d).', 3*L, numel(params));
    end
    if numel(pi) ~= n
        error('recover_rbfs_approximations:badPi', ...
              'Permutation pi must have length n (= %d).', n);
    end

    % 1) Build the SAME grid of centres as in reorder_fit_rbfs
    g        = ceil(sqrt(L));                 % square-ish grid
    cx       = linspace(1, n, g);
    cy       = linspace(1, max(n,m), g);
    [CX, CY] = meshgrid(cx, cy);
    centres  = [CX(:), CY(:)];
    centres  = centres(1:L, :);               % keep first L

    % 2) Split weights and evaluate permuted-space surfaces
    a = params(1:L);            % A1 weights
    b = params(L+1:2*L);        % A2 weights
    c = params(2*L+1:3*L);      % A3 weights

    F1_perm = eval_rbf2d(a, n, n, centres, sigma);
    F2_perm = eval_rbf2d(b, n, m, centres, sigma);
    F3_perm = eval_rbf2d(c, n, m, centres, sigma);

    % 3) Invert permutation to map back to ORIGINAL order
    invpi = inverse_permutation(pi);

    A1_hat = F1_perm(invpi, invpi);   % rows & cols
    A2_hat = F2_perm(invpi, :);       % rows only
    A3_hat = F3_perm(invpi, :);       % rows only
end

% -------------------------------------------------------------------------
function Z = eval_rbf2d(weights, rows, cols, centres, sigma)
% Evaluate Σ_k w_k exp(-||[x,y]-c_k||^2 / (2σ^2)) on a rows×cols grid.
    [X, Y] = meshgrid(1:cols, 1:rows);    % X: col index, Y: row index
    Z      = zeros(rows, cols);
    L      = numel(weights);
    for k = 1:L
        dx2 = (X - centres(k,1)).^2 + (Y - centres(k,2)).^2;
        Z   = Z + weights(k) .* exp(-dx2 / (2*sigma^2));
    end
end

% -------------------------------------------------------------------------
function invpi = inverse_permutation(pi)
% invpi(pi(k)) = k
    n     = numel(pi);
    invpi = zeros(size(pi));
    invpi(pi) = 1:n;
end


