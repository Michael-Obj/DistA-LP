function violRatio = violation_ratio(D_true, D_hat)
%VIOLATION_RATIO Fraction of entries where D_hat > D_true
%
%   violRatio = violation_ratio(D_true, D_hat)
%
% Inputs:
%   D_true : true distance matrix (n×m)
%   D_hat  : approximated distance matrix (n×m)
%
% Output:
%   violRatio : fraction of entries with D_hat > D_true

    if ~isequal(size(D_true), size(D_hat))
        error('D_true and D_hat must have the same size.');
    end

    numViolations = sum(D_hat(:) > D_true(:));
    totalEntries  = numel(D_true);

    violRatio = numViolations / totalEntries;
end