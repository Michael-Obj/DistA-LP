function [A1_hat, A2_hat, A3_hat] = recover_polynomial_approximations(n, m, coeffs, deg, pi)
% Recover approximated A1 (n×n), A2 (n×m), A3 (n×m) in the ORIGINAL order
% from polynomial coefficients and the (row) permutation pi used during fitting.
%
% Inputs
%   n, m    : target sizes (A1 is n×n; A2,A3 are n×m)
%   coeffs  : row vector of concatenated polynomial coeffs
%             [c1(1..L), c2(1..L), c3(1..L)], L=(deg+1)(deg+2)/2
%   deg     : polynomial degree used in fitting (e.g., 3)
%   pi      : permutation indices (length n) applied during fitting
%
% Outputs
%   A1_hat  : n×n recovered approximation of A1 in the ORIGINAL order
%   A2_hat  : n×m recovered approximation of A2 in the ORIGINAL order
%   A3_hat  : n×m recovered approximation of A3 in the ORIGINAL order

    % ---- 1) Rebuild fitted polynomial “surfaces” in the permuted space ----
    [F1_perm, F2_perm, F3_perm] = generate_polynomials_recover(n, m, coeffs, deg);

    % ---- 2) Invert permutation to map back to original index order ----
    invpi = inverse_permutation(pi);

    % ---- 3) Unpermute rows/cols back to ORIGINAL order ----
    A1_hat = F1_perm(invpi, invpi);  % rows & cols
    A2_hat = F2_perm(invpi,   :);    % rows only
    A3_hat = F3_perm(invpi,   :);    % rows only
end

% -------------------------------------------------------------------------
function invpi = inverse_permutation(pi)
% Return inverse permutation such that invpi(pi(k)) = k
    n    = numel(pi);
    invpi = zeros(size(pi));
    invpi(pi) = 1:n;
end

% -------------------------------------------------------------------------
function [F1, F2, F3] = generate_polynomials_recover(n, m, coeffs, deg)
% Build n×n and n×m polynomial surfaces from concatenated coefficients.

    L = (deg + 1)*(deg + 2)/2;

    if numel(coeffs) ~= 3*L
        error('recover_polynomial_approximations:badCoeffLength', ...
              'Expected coeffs to have length %d (got %d).', 3*L, numel(coeffs));
    end

    c1 = coeffs(1:L);
    c2 = coeffs(L+1:2*L);
    c3 = coeffs(2*L+1:3*L);

    % IMPORTANT: meshgrid(X, Y) -> size(Y)×size(X)
    % For n×n:
    [X1, Y1] = meshgrid(1:n, 1:n);   % -> n×n
    % For n×m:
    [X2, Y2] = meshgrid(1:m, 1:n);   % -> n×m

    F1 = eval_poly2d_coeffs(c1, X1, Y1, deg);   % n×n
    F2 = eval_poly2d_coeffs(c2, X2, Y2, deg);   % n×m
    F3 = eval_poly2d_coeffs(c3, X2, Y2, deg);   % n×m
end

% -------------------------------------------------------------------------
function Z = eval_poly2d_coeffs(c, X, Y, deg)
% Evaluate a total-degree-`deg` 2-D polynomial with coeff vector c
% on the coordinate grids X (cols), Y (rows).
%
% Basis ordering matches reorder_fit_polynomials:
%     for p = 0:deg
%       for q = 0:(deg-p)
%           k = k+1;  Z += c(k) * X.^p .* Y.^q;

    Z = zeros(size(X));
    k = 0;
    for p = 0:deg
        for q = 0:(deg - p)
            k  = k + 1;
            Z  = Z + c(k) .* (X.^p) .* (Y.^q);
        end
    end
end

