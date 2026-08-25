function estimate = estimate_lp_resources(method, n, m, r)
%ESTIMATE_LP_RESOURCES Estimate the main sparse LP setup allocations.

    if nargin < 4
        r = 0;
    end
    method = upper(string(method));
    pair_rows = m * n * (n - 1);

    if method == "LP"
        variable_count = n * m;
        inequality_rows = pair_rows;
        inequality_nnz = 2 * pair_rows;
        equality_rows = n;
        equality_nnz = n * m;
    elseif method == "COPT"
        r = min(max(round(r), 1), n);
        variable_count = n * m + m + 1;
        inequality_rows = pair_rows + 2 * n;
        inequality_nnz = 2 * pair_rows + 2 * n * m + n;
        equality_rows = m * (n - r);
        equality_nnz = 2 * equality_rows;
    else
        error('Baseline:UnknownMethod', 'Unknown LP method: %s', method);
    end

    % MATLAB sparse matrices store roughly one double value and one index
    % per nonzero, plus a column pointer. Construction also holds three
    % double triplet arrays (row, column, value) and RHS vectors.
    sparse_a_bytes = 16 * inequality_nnz + 8 * (variable_count + 1);
    sparse_aeq_bytes = 16 * equality_nnz + 8 * (variable_count + 1);
    triplet_bytes = 24 * (inequality_nnz + equality_nnz);
    rhs_bytes = 8 * (inequality_rows + equality_rows);
    vector_bytes = 8 * (3 * variable_count);
    estimated_peak_bytes = sparse_a_bytes + sparse_aeq_bytes + ...
        triplet_bytes + rhs_bytes + vector_bytes;

    estimate = struct();
    estimate.method = char(method);
    estimate.real_records = n;
    estimate.outputs = m;
    estimate.variables = variable_count;
    estimate.inequality_rows = inequality_rows;
    estimate.equality_rows = equality_rows;
    estimate.nonzeros = inequality_nnz + equality_nnz;
    estimate.estimated_peak_bytes = estimated_peak_bytes;
end
