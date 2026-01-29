function [A1_hat, A2_hat, A3_hat] = recover_gaussian_approximations(n, m, params, pi)
% Recover approximated A1 (n×n), A2 (n×m), A3 (n×m) in the ORIGINAL order
% from Gaussian parameters and the (row) permutation pi used during fitting.

    % 1) Generate fitted surfaces in the permuted space
    [F1_perm, F2_perm, F3_perm] = generate_gaussians_recover(n, m, params);

    % 2) Invert the permutation to map back to original index order
    invpi = inverse_permutation(pi);

    % 3) Unpermute rows/cols
    A1_hat = F1_perm(invpi, invpi);   % rows & cols
    A2_hat = F2_perm(invpi, :);       % rows only
    A3_hat = F3_perm(invpi, :);       % rows only
end

function invpi = inverse_permutation(pi)
    n = numel(pi);
    invpi = zeros(size(pi));
    invpi(pi) = 1:n;
end

function [F1, F2, F3] = generate_gaussians_recover(n, m, params)
% Build n×n and n×m Gaussian surfaces from parameters.

    % IMPORTANT: meshgrid(X, Y) -> size(Y)×size(X)
    % For n×n:
    [X1, Y1] = meshgrid(1:n, 1:n);      % -> n×n
    % For n×m:
    [X2, Y2] = meshgrid(1:m, 1:n);      % -> n×m  (columns 1..m, rows 1..n)

    p1 = params(1:6);
    p2 = params(7:12);
    p3 = params(13:18);

    F1 = gaussian2d(X1, Y1, p1(1), p1(2), p1(3), p1(4), p1(5), p1(6)); % n×n
    F2 = gaussian2d(X2, Y2, p2(1), p2(2), p2(3), p2(4), p2(5), p2(6)); % n×m
    F3 = gaussian2d(X2, Y2, p3(1), p3(2), p3(3), p3(4), p3(5), p3(6)); % n×m
end

function val = gaussian2d(x, y, cx, cy, sx, sy, A, b)
% A·exp(-((x-cx)^2/(2sx^2) + (y-cy)^2/(2sy^2))) + b
    exponent = -((x - cx).^2 ./ (2 * sx^2) + (y - cy).^2 ./ (2 * sy^2));
    val = A * exp(exponent) + b;
end
