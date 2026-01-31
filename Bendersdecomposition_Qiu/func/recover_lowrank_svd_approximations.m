function [A1_hat, A2_hat, A3_hat] = recover_lowrank_svd_approximations(n, m, factors, pi)
% Recover approximated A1 (n×n), A2 (n×m), A3 (n×m) in the ORIGINAL order
% from truncated-SVD factors and the (row) permutation `pi` used during fitting.
%
% Inputs
%   n, m     : sizes (A1 is n×n; A2,A3 are n×m)
%   factors  : struct with fields U1,S1,V1,U2,S2,V2,U3,S3,V3 (from reorder_fit_lowrank_svd)
%   pi       : 1×n permutation applied during fitting
%
% Outputs
%   A1_hat, A2_hat, A3_hat : recovered matrices in ORIGINAL row/column order

    % ---- sanity checks ----
    req = {'U1','S1','V1','U2','S2','V2','U3','S3','V3'};
    for k = 1:numel(req)
        if ~isfield(factors, req{k})
            error('recover_lowrank_svd_approximations:missingField', ...
                  'Missing field "%s" in factors.', req{k});
        end
    end
    if numel(pi) ~= n
        error('recover_lowrank_svd_approximations:badPi', ...
              'Permutation pi must have length n (= %d).', n);
    end

    % ---- 1) Rebuild fitted surfaces in the permuted space ----
    F1_perm = factors.U1 * factors.S1 * factors.V1';   % n×n
    F2_perm = factors.U2 * factors.S2 * factors.V2';   % n×m
    F3_perm = factors.U3 * factors.S3 * factors.V3';   % n×m

    % ---- 2) Invert permutation to map back to ORIGINAL index order ----
    invpi = inverse_permutation(pi);

    % ---- 3) Unpermute rows/cols ----
    A1_hat = F1_perm(invpi, invpi);   % rows & cols
    A2_hat = F2_perm(invpi, :);       % rows only
    A3_hat = F3_perm(invpi, :);       % rows only
end

% -------------------------------------------------------------------------
function invpi = inverse_permutation(pi)
% invpi(pi(k)) = k
    n     = numel(pi);
    invpi = zeros(size(pi));
    invpi(pi) = 1:n;
end

