function results = run_optimization_baseline(method, config)
%RUN_OPTIMIZATION_BASELINE Common runner for COPT and centralized LP.
%
% The returned results.raw table is the programmatic interface. Primary
% runner scripts also print paper-style summaries and do not save files.

    method = upper(string(method));
    if method ~= "COPT" && method ~= "LP"
        error('Baseline:UnknownMethod', 'Method must be COPT or LP.');
    end
    config = normalize_config(config);
    repeat_ids = resolve_repeats(config);

    raw = table();
    fprintf('\n%s baseline started: city=%s, records=%d\n', ...
        method, upper(config.city), config.node_count);
    if isempty(config.record_limit)
        fprintf('Domain mode: full supplied domain\n');
    else
        fprintf('Domain mode: interface smoke test with at most %d records\n', ...
            config.record_limit);
    end
    fprintf('Solver limit: %.0f seconds; setup safety limit: %.3g GB\n', ...
        config.max_time_seconds, config.max_setup_bytes / 1e9);

    for user_id = config.user_ids
        for repeat_id = repeat_ids
            try
                metadata = inspect_baseline_case(config.city, ...
                    config.node_count, user_id, repeat_id, config.record_limit);
                if method == "COPT"
                    preflight = estimate_lp_resources('COPT', ...
                        metadata.effective_record_count, metadata.output_count, ...
                        config.copt_neighbors);
                else
                    preflight = estimate_lp_resources('LP', ...
                        metadata.effective_record_count, metadata.output_count);
                end
                if ~config.allow_oversized && ...
                        preflight.estimated_peak_bytes > config.max_setup_bytes
                    detail = sprintf( ...
                        ['Estimated %s setup is %.3g GB, above the configured ' ...
                         '%.3g GB safety limit. Set allow_oversized=true to force an attempt.'], ...
                        method, preflight.estimated_peak_bytes / 1e9, ...
                        config.max_setup_bytes / 1e9);
                    for epsilon = config.epsilons
                        raw = append_row(raw, config, method, user_id, repeat_id, ...
                            epsilon, metadata.effective_record_count, ...
                            "estimated_infeasible", NaN, NaN, NaN, NaN, ...
                            preflight.estimated_peak_bytes, detail);
                        fprintf('  user=%d repeat=%d epsilon=%g: estimated_infeasible\n', ...
                            user_id, repeat_id, epsilon);
                    end
                    continue;
                end

                case_data = load_baseline_case(config.city, config.node_count, ...
                    user_id, repeat_id, config.record_limit);
            catch ME
                for epsilon = config.epsilons
                    raw = append_row(raw, config, method, user_id, repeat_id, ...
                        epsilon, config.node_count, "failed", NaN, NaN, NaN, ...
                        NaN, NaN, string(ME.message));
                end
                continue;
            end

            for epsilon = config.epsilons
                solver_config = struct( ...
                    'max_time_seconds', config.max_time_seconds, ...
                    'max_setup_bytes', config.max_setup_bytes, ...
                    'allow_oversized', config.allow_oversized, ...
                    'display', config.solver_display);

                if method == "COPT"
                    [mechanism, utility_loss, runtime_sec, solve_result] = ...
                        solve_copt_baseline(case_data.distance_ro, ...
                            case_data.distance_rr, case_data.cost_matrix, ...
                            epsilon, config.copt_lambda, ...
                            config.copt_neighbors, solver_config);
                else
                    [mechanism, utility_loss, runtime_sec, solve_result] = ...
                        solve_metric_dp_lp(case_data.distance_rr, ...
                            case_data.cost_matrix, epsilon, solver_config);
                end

                if solve_result.status == "optimal"
                    violation_ratio = metric_dp_violation_ratio( ...
                        mechanism, case_data.distance_rr, epsilon, ...
                        config.violation_tolerance);
                else
                    violation_ratio = NaN;
                end

                raw = append_row(raw, config, method, user_id, repeat_id, ...
                    epsilon, case_data.effective_node_count, ...
                    solve_result.status, utility_loss, violation_ratio, ...
                    runtime_sec, solve_result.exitflag, ...
                    solve_result.estimate.estimated_peak_bytes, ...
                    solve_result.message);
                fprintf('  user=%d repeat=%d epsilon=%g: %s\n', ...
                    user_id, repeat_id, epsilon, solve_result.status);
            end
        end
    end

    summary = summarize_results(raw, method, config, repeat_ids);
    print_summary(summary, raw, method, config, repeat_ids);
    results = struct('method', method, 'config', config, ...
        'raw', raw, 'summary', summary);
end

function raw = append_row(raw, config, method, user_id, repeat_id, ...
        epsilon, effective_records, status, utility_loss, violation_ratio, ...
        runtime_sec, exitflag, estimated_setup_bytes, detail)
    row = table(string(config.city), config.node_count, effective_records, ...
        user_id, repeat_id, epsilon, method, string(status), utility_loss, ...
        violation_ratio, runtime_sec, exitflag, estimated_setup_bytes, ...
        string(detail), 'VariableNames', ...
        {'city','node_count','effective_records','user_id','repeat_id', ...
         'epsilon','method','status','utility_loss','violation_ratio', ...
         'runtime_sec','exitflag','estimated_setup_bytes','detail'});
    raw = [raw; row];
end

function summary = summarize_results(raw, method, config, repeat_ids)
    nr_eps = numel(config.epsilons);
    utility_mean = nan(1, nr_eps);
    utility_std = nan(1, nr_eps);
    runtime_mean = nan(1, nr_eps);
    runtime_std = nan(1, nr_eps);
    violation_mean = nan(1, nr_eps);
    violation_std = nan(1, nr_eps);
    statuses = strings(1, nr_eps);

    for eps_idx = 1:nr_eps
        epsilon = config.epsilons(eps_idx);
        subset = raw(raw.epsilon == epsilon & raw.method == method, :);
        statuses(eps_idx) = join(unique(subset.status, 'stable'), ',');
        utility_by_repeat = [];
        runtime_by_repeat = [];
        violation_by_repeat = [];

        for repeat_id = repeat_ids
            repeat_rows = subset(subset.repeat_id == repeat_id, :);
            if height(repeat_rows) ~= numel(config.user_ids) || ...
                    any(repeat_rows.status ~= "optimal")
                continue;
            end
            utility_by_repeat(end + 1) = sum(repeat_rows.utility_loss); %#ok<AGROW>
            runtime_by_repeat(end + 1) = mean(repeat_rows.runtime_sec); %#ok<AGROW>
            violation_by_repeat(end + 1) = mean(repeat_rows.violation_ratio); %#ok<AGROW>
        end

        if numel(utility_by_repeat) == numel(repeat_ids)
            utility_mean(eps_idx) = mean(utility_by_repeat) / 10000;
            utility_std(eps_idx) = std(utility_by_repeat) / 10000;
            runtime_mean(eps_idx) = mean(runtime_by_repeat);
            runtime_std(eps_idx) = std(runtime_by_repeat);
            violation_mean(eps_idx) = mean(violation_by_repeat);
            violation_std(eps_idx) = std(violation_by_repeat);
        end
    end

    summary = table(config.epsilons(:), utility_mean(:), utility_std(:), ...
        runtime_mean(:), runtime_std(:), violation_mean(:), ...
        violation_std(:), statuses(:), 'VariableNames', ...
        {'epsilon','utility_mean','utility_std','runtime_mean','runtime_std', ...
         'violation_mean','violation_std','statuses'});
end

function print_summary(summary, raw, method, config, repeat_ids)
    fprintf('\nBaseline setting: city=%s, records=%d, users=%d, repetitions=%d\n', ...
        upper(config.city), config.node_count, numel(config.user_ids), ...
        numel(repeat_ids));
    print_metric('Utility loss (10,000 meters)', method, summary, ...
        'utility_mean', 'utility_std', 2);
    print_metric('Violation ratio', method, summary, ...
        'violation_mean', 'violation_std', 4);
    print_metric('Computation time (seconds)', method, summary, ...
        'runtime_mean', 'runtime_std', 4);

    fprintf('\nExecution status\n');
    fprintf('%-12s', 'Method');
    for eps_idx = 1:height(summary)
        fprintf(' | %-20s', sprintf('epsilon=%g', summary.epsilon(eps_idx)));
    end
    fprintf('\n%s\n', repmat('-', 1, 12 + 23 * height(summary)));
    fprintf('%-12s', method);
    for eps_idx = 1:height(summary)
        status_text = char(summary.statuses(eps_idx));
        if isempty(status_text)
            status_text = 'no cases';
        end
        fprintf(' | %-20s', status_text);
    end
    fprintf('\n');

    estimates = raw.estimated_setup_bytes(~isnan(raw.estimated_setup_bytes));
    if ~isempty(estimates)
        fprintf('Largest estimated setup allocation: %.3g GB\n', max(estimates) / 1e9);
    end
    if any(raw.status == "estimated_infeasible")
        fprintf(['No utility or violation value is fabricated for preflighted cases. ' ...
            'Set allow_oversized=true only when an actual full-scale attempt is intended.\n']);
    end
end

function print_metric(title_text, method, summary, mean_field, std_field, digits)
    fprintf('\n%s -- mean +/- standard deviation\n', title_text);
    fprintf('%-12s', 'Method');
    for eps_idx = 1:height(summary)
        fprintf(' | epsilon=%-4g', summary.epsilon(eps_idx));
    end
    fprintf('\n%s\n', repmat('-', 1, 12 + 16 * height(summary)));
    fprintf('%-12s', method);
    value_format = sprintf(' | %%.%df +/- %%.%df', digits, digits);
    for eps_idx = 1:height(summary)
        mean_value = summary.(mean_field)(eps_idx);
        std_value = summary.(std_field)(eps_idx);
        if isnan(mean_value)
            fprintf(' | %-12s', '------------');
        else
            fprintf(value_format, mean_value, std_value);
        end
    end
    fprintf('\n');
end

function repeat_ids = resolve_repeats(config)
    if ~isempty(config.repeat_ids)
        repeat_ids = config.repeat_ids;
    elseif strcmpi(config.city, 'london')
        repeat_ids = 1:6;
    elseif strcmpi(config.city, 'rome') && config.node_count == 2000
        repeat_ids = 1:5;
    else
        repeat_ids = 1:4;
    end
end

function config = normalize_config(config)
    required = {'city','node_count','epsilons','user_ids'};
    for idx = 1:numel(required)
        if ~isfield(config, required{idx})
            error('Baseline:MissingConfig', ...
                'Missing required configuration field: %s', required{idx});
        end
    end
    if ~isfield(config, 'repeat_ids'); config.repeat_ids = []; end
    if ~isfield(config, 'record_limit'); config.record_limit = []; end
    if ~isfield(config, 'max_time_seconds'); config.max_time_seconds = 1800; end
    if ~isfield(config, 'max_setup_bytes'); config.max_setup_bytes = 4 * 1024^3; end
    if ~isfield(config, 'allow_oversized'); config.allow_oversized = false; end
    if ~isfield(config, 'solver_display'); config.solver_display = 'off'; end
    if ~isfield(config, 'violation_tolerance'); config.violation_tolerance = 1e-8; end
    if ~isfield(config, 'copt_lambda'); config.copt_lambda = 100; end
    if ~isfield(config, 'copt_neighbors'); config.copt_neighbors = 5; end
end
