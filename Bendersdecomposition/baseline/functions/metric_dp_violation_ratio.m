function ratio = metric_dp_violation_ratio( ...
        mechanism, distance_rr, epsilon, tolerance)
%METRIC_DP_VIOLATION_RATIO Fraction of ordered mDP constraints violated.

    if nargin < 4
        tolerance = 1e-8;
    end
    [n, m] = size(mechanism);
    if isempty(mechanism)
        ratio = NaN;
        return;
    end

    privacy_factor = exp(epsilon * distance_rr);
    off_diagonal = ~eye(n);
    violation_count = 0;
    for output_idx = 1:m
        probabilities = mechanism(:, output_idx);
        allowed = privacy_factor .* probabilities';
        violated = probabilities > allowed + tolerance;
        violation_count = violation_count + nnz(violated & off_diagonal);
    end
    ratio = violation_count / (m * n * (n - 1));
end
