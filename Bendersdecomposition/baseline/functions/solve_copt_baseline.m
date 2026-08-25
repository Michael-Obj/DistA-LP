function [mechanism, utility_loss, runtime_sec, result] = ...
        solve_copt_baseline(distance_ro, distance_rr, cost_matrix, ...
        epsilon, lambda, neighbor_count, solver_config)
%SOLVE_COPT_BASELINE Validate, preflight, and invoke supplied COPT code.

    if nargin < 7
        solver_config = struct();
    end
    solver_config = normalize_config(solver_config);
    [n, m] = size(distance_ro);
    validate_inputs(distance_ro, distance_rr, cost_matrix, epsilon, n, m);
    neighbor_count = min(round(neighbor_count), n);
    estimate = estimate_lp_resources('COPT', n, m, neighbor_count);

    result = struct('method', 'COPT', 'status', "not_started", ...
        'message', "", 'exitflag', NaN, 'solver_output', struct(), ...
        'error_identifier', "", 'estimate', estimate);
    mechanism = [];
    utility_loss = NaN;
    runtime_sec = NaN;

    if ~solver_config.allow_oversized && ...
            estimate.estimated_peak_bytes > solver_config.max_setup_bytes
        result.status = "estimated_infeasible";
        result.message = sprintf( ...
            ['Estimated COPT setup is %.3g GB, above the configured %.3g GB ' ...
             'safety limit. Set allow_oversized=true to force an attempt.'], ...
            estimate.estimated_peak_bytes / 1e9, ...
            solver_config.max_setup_bytes / 1e9);
        return;
    end

    start_time = tic;
    try
        copt_options = struct('max_time_seconds', ...
            solver_config.max_time_seconds, 'display', solver_config.display);
        [mechanism, raw_mechanism, utility_loss, exitflag, output] = ...
            compute_copt2(distance_ro, distance_rr, cost_matrix, ...
                epsilon, lambda, neighbor_count, copt_options); %#ok<ASGLU>
        runtime_sec = toc(start_time);
        result.exitflag = exitflag;
        result.solver_output = output;
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

function validate_inputs(distance_ro, distance_rr, cost_matrix, epsilon, n, m)
    if ~isequal(size(distance_rr), [n, n])
        error('Baseline:InvalidDistanceMatrix', ...
            'distance_rr must be n-by-n and match distance_ro.');
    end
    if ~isequal(size(cost_matrix), [n, m])
        error('Baseline:InvalidCostMatrix', ...
            'cost_matrix must have the same size as distance_ro.');
    end
    if any(~isfinite(distance_ro), 'all') || any(distance_ro < 0, 'all') || ...
            any(~isfinite(distance_rr), 'all') || any(distance_rr < 0, 'all') || ...
            any(~isfinite(cost_matrix), 'all') || any(cost_matrix < 0, 'all')
        error('Baseline:InvalidInput', ...
            'COPT distance and cost inputs must be finite and nonnegative.');
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
