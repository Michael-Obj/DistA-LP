function [mechanism, utility_loss, runtime_sec, result] = ...
        solve_metric_dp_lp(distance_rr, cost_matrix, epsilon, solver_config)
%SOLVE_METRIC_DP_LP Solve the original centralized metric-DP LP.
%
% Variables are z(i,k), the probability of output k for input i.
% The LP minimizes expected utility loss subject to row normalization and
% every pairwise metric-DP constraint
%   z(i,k) <= exp(epsilon*d(i,j))*z(j,k).

    if nargin < 4
        solver_config = struct();
    end
    solver_config = normalize_config(solver_config);
    [n, m] = size(cost_matrix);
    validate_inputs(distance_rr, cost_matrix, epsilon, n);

    estimate = estimate_lp_resources('LP', n, m);
    result = initial_result('LP', estimate);
    mechanism = [];
    utility_loss = NaN;
    runtime_sec = NaN;

    if ~solver_config.allow_oversized && ...
            estimate.estimated_peak_bytes > solver_config.max_setup_bytes
        result.status = "estimated_infeasible";
        result.message = sprintf( ...
            ['Estimated LP setup is %.3g GB, above the configured %.3g GB ' ...
             'safety limit. Set allow_oversized=true to force an attempt.'], ...
            estimate.estimated_peak_bytes / 1e9, ...
            solver_config.max_setup_bytes / 1e9);
        return;
    end

    start_time = tic;
    try
        pair_mask = ~eye(n);
        [input_i, input_j] = find(pair_mask);
        pair_count = numel(input_i);
        constraint_count = m * pair_count;
        nonzero_count = 2 * constraint_count;

        row_idx = zeros(nonzero_count, 1);
        col_idx = zeros(nonzero_count, 1);
        values = zeros(nonzero_count, 1);
        pair_linear = sub2ind([n, n], input_i, input_j);
        privacy_factors = exp(epsilon * distance_rr(pair_linear));

        cursor = 1;
        for output_idx = 1:m
            block_rows = ((output_idx - 1) * pair_count + (1:pair_count))';
            block = cursor:(cursor + 2 * pair_count - 1);
            row_idx(block(1:2:end)) = block_rows;
            row_idx(block(2:2:end)) = block_rows;
            offset = (output_idx - 1) * n;
            col_idx(block(1:2:end)) = input_i + offset;
            col_idx(block(2:2:end)) = input_j + offset;
            values(block(1:2:end)) = 1;
            values(block(2:2:end)) = -privacy_factors;
            cursor = cursor + 2 * pair_count;
        end

        A = sparse(row_idx, col_idx, values, constraint_count, n * m);
        b = zeros(constraint_count, 1);
        Aeq = kron(ones(1, m), speye(n));
        beq = ones(n, 1);
        f = cost_matrix(:);
        lb = zeros(n * m, 1);
        ub = ones(n * m, 1);
        clear row_idx col_idx values input_i input_j pair_linear privacy_factors

        setup_time = toc(start_time);
        remaining_time = solver_config.max_time_seconds - setup_time;
        if remaining_time <= 0
            runtime_sec = setup_time;
            result.status = "time_limit";
            result.message = 'The 1,800-second budget expired during LP setup.';
            return;
        end

        options = optimoptions('linprog', ...
            'Display', solver_config.display, ...
            'MaxTime', remaining_time, ...
            MaxIterations=5e10,...
        Algorithm="interior-point");
        [x, ~, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, [], options);
        runtime_sec = toc(start_time);
        result.exitflag = exitflag;
        result.solver_output = output;

        if ~isempty(x)
            mechanism = reshape(x, n, m);
            utility_loss = sum(cost_matrix .* mechanism, 'all');
        end
        result.status = map_exitflag(exitflag, runtime_sec, ...
            solver_config.max_time_seconds);
        result.message = sprintf('linprog exitflag=%d; status=%s', ...
            exitflag, result.status);
    catch ME
        runtime_sec = toc(start_time);
        result.status = "failed";
        result.message = string(ME.message);
        result.error_identifier = string(ME.identifier);
    end
end

function validate_inputs(distance_rr, cost_matrix, epsilon, n)
    if ~isequal(size(distance_rr), [n, n])
        error('Baseline:InvalidDistanceMatrix', ...
            'distance_rr must be n-by-n and match cost_matrix.');
    end
    if any(~isfinite(distance_rr), 'all') || any(distance_rr < 0, 'all')
        error('Baseline:InvalidDistanceMatrix', ...
            'distance_rr must contain finite, nonnegative values.');
    end
    if any(~isfinite(cost_matrix), 'all') || any(cost_matrix < 0, 'all')
        error('Baseline:InvalidCostMatrix', ...
            'cost_matrix must contain finite, nonnegative values.');
    end
    if ~isscalar(epsilon) || ~isfinite(epsilon) || epsilon <= 0
        error('Baseline:InvalidEpsilon', 'epsilon must be positive.');
    end
end

function config = normalize_config(config)
    if ~isfield(config, 'max_time_seconds'); config.max_time_seconds = 1800; end
    if ~isfield(config, 'max_setup_bytes'); config.max_setup_bytes = 4 * 1024^3; end
    if ~isfield(config, 'allow_oversized'); config.allow_oversized = false; end
    if ~isfield(config, 'display'); config.display = 'off'; end
end

function result = initial_result(method, estimate)
    result = struct('method', method, 'status', "not_started", ...
        'message', "", 'exitflag', NaN, 'solver_output', struct(), ...
        'error_identifier', "", 'estimate', estimate);
end

function status = map_exitflag(exitflag, runtime_sec, max_time_seconds)
    if exitflag == 1
        status = "optimal";
    elseif exitflag == 0 || runtime_sec >= max_time_seconds
        status = "time_limit";
    elseif exitflag == -2
        status = "infeasible";
    elseif exitflag == -3
        status = "unbounded";
    else
        status = "failed";
    end
end
