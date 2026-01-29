function relErr = relative_error(A, A_hat)
%RELATIVE_FROBENIUS_ERROR Compute the relative Frobenius norm error
%
%   relErr = relative_frobenius_error(A, A_hat)
%
% Inputs:
%   A      : Original matrix (n×m)
%   A_hat  : Approximated matrix (n×m)
%
% Output:
%   relErr : Relative Frobenius norm error

    if ~isequal(size(A), size(A_hat))
        error('A and A_hat must have the same size.');
    end

    numerator = norm(A - A_hat, 'fro');
    denominator = norm(A, 'fro');

    relErr = numerator / denominator;
end
