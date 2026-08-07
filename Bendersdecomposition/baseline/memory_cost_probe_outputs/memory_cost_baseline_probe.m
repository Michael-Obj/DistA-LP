% Memory-cost probe for SP_baseline.
% Run from MATLAB with:
%   run('memory_cost_probe_outputs/memory_cost_baseline_probe.m')

clear;
clc;
rng(1);

script_dir = fileparts(mfilename('fullpath'));
cd(fullfile(script_dir, '..'));
base_dir = '.';
result_dir = fullfile('memory_cost_probe_outputs');

cfg = struct();
cfg.city = 'london';
cfg.node_count = 2000;
cfg.user_id = 1;
cfg.repeat_id = 1;
cfg.epsilon = 4;
cfg.run_em_embr = true;
cfg.run_lpca = true;

cfg = apply_env_override(cfg, 'city', 'MEMPROBE_CITY', true);
cfg = apply_env_override(cfg, 'node_count', 'MEMPROBE_NODE_COUNT', false);
cfg = apply_env_override(cfg, 'user_id', 'MEMPROBE_USER_ID', false);
cfg = apply_env_override(cfg, 'repeat_id', 'MEMPROBE_REPEAT_ID', false);
cfg = apply_env_override(cfg, 'epsilon', 'MEMPROBE_EPSILON', false);
cfg = apply_env_bool_override(cfg, 'run_em_embr', 'MEMPROBE_RUN_EM_EMBR');
cfg = apply_env_bool_override(cfg, 'run_lpca', 'MEMPROBE_RUN_LPCA');

addpath(fullfile(base_dir, 'functions'));
addpath(fullfile(base_dir, 'functions', 'haversine'));
addpath(fullfile(base_dir, 'functions', 'myBDToolbox'));
addpath(fullfile(base_dir, 'functions', 'myPlotToolbox'));
addpath(fullfile(base_dir, 'functions', 'myRLToolbox'));

records = table();
status = "ok";
error_message = "";
started_at = datetime("now");

try
    fprintf('Memory probe: %s, %d nodes, user %d, repeat %d, epsilon %.3g\n', ...
        cfg.city, cfg.node_count, cfg.user_id, cfg.repeat_id, cfg.epsilon);

    data_file = fullfile(base_dir, sprintf('%s_location_data_%d_nodes', cfg.city, cfg.node_count), ...
        sprintf('location_data_sample_%d', cfg.user_id), ...
        sprintf('location_data_r%d_user%d.mat', cfg.repeat_id, cfg.user_id));
    if ~isfile(data_file)
        error('Data file not found: %s', data_file);
    end
    load(data_file);

    raw_dir = fullfile(base_dir, 'Dataset', cfg.city, 'raw');
    node_csv = fullfile(raw_dir, sprintf('%s_nodes.csv', cfg.city));
    edge_csv = fullfile(raw_dir, sprintf('%s_edges.csv', cfg.city));
    if strcmp(cfg.city, 'nyc')
        node_csv = fullfile(raw_dir, 'nyc_nodes.csv');
        edge_csv = fullfile(raw_dir, 'nyc_edges.csv');
    end
    opts = detectImportOptions(node_csv);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_csv, opts);
    df_edges = readtable(edge_csv);

    col_longitude = table2array(df_nodes(:, 'x'));
    col_latitude = table2array(df_nodes(:, 'y'));
    parameters;
    env_parameters.EPSILON = cfg.epsilon;
    env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
    env_parameters.NR_OBFLOC = numel(obf_ID);

    load(fullfile(base_dir, sprintf('G_%s.mat', cfg.city)), 'G');
    if isfile(fullfile(base_dir, sprintf('u_%s.mat', cfg.city)))
        load(fullfile(base_dir, sprintf('u_%s.mat', cfg.city)), 'u');
    end
    if isfile(fullfile(base_dir, sprintf('v_%s.mat', cfg.city)))
        load(fullfile(base_dir, sprintf('v_%s.mat', cfg.city)), 'v');
    end

    records = append_record(records, "after_load", cfg, NaN, NaN, NaN, NaN, "loaded inputs");

    distance_matrix = distanceMatrix(col_longitude(node_tar), col_latitude(node_tar));
    distance_save = distance_matrix;
    task_loc = 2;
    records = append_record(records, "after_distance_matrix", cfg, NaN, NaN, NaN, NaN, "full target distance matrix built");

    if cfg.run_em_embr
        fprintf('Running EM/EMBR...\n');
        t0 = tic;
        [loss_EM, loss_EMBR, P_matrix, time_EM, time_BR] = loss_for_benchmark( ...
            env_parameters, obf_ID, distance_matrix, node_tar, G, task_loc);
        runtime = toc(t0);
        detail = sprintf('loss_EM=%.8g; loss_EMBR=%.8g; time_EM=%.4g; time_BR=%.4g; P_matrix_bytes=%d', ...
            loss_EM, loss_EMBR, time_EM, time_BR, bytes_of_var(P_matrix));
        records = append_record(records, "after_em_embr", cfg, runtime, loss_EM, loss_EMBR, bytes_of_var(P_matrix), detail);
    end

    lpca_est = estimate_lpca_matrix_bytes(col_longitude, col_latitude, obf_ID);
    detail = sprintf('LPCA dense A estimate: NR_RECORD=%d; NR_OBF=%d; A_rows=%g; A_cols=%g; A_bytes=%g', ...
        lpca_est.NR_RECORD, lpca_est.NR_OBF, lpca_est.A_rows, lpca_est.A_cols, lpca_est.A_bytes);
    records = append_record(records, "before_lpca", cfg, NaN, NaN, NaN, lpca_est.A_bytes, detail);

    full_lp_est = estimate_full_lp_matrix_bytes(numel(node_tar), numel(obf_ID));
    detail = sprintf('Full dense LP A estimate: NR_RECORD=%d; NR_OBF=%d; A_rows=%g; A_cols=%g; A_bytes=%g', ...
        full_lp_est.NR_RECORD, full_lp_est.NR_OBF, full_lp_est.A_rows, full_lp_est.A_cols, full_lp_est.A_bytes);
    records = append_record(records, "full_lp_estimate_only", cfg, NaN, NaN, NaN, full_lp_est.A_bytes, detail);

    if cfg.run_lpca
        fprintf('Running LPCA/coarse LP...\n');
        t0 = tic;
        coarse;
        runtime = toc(t0);
        A_bytes = bytes_if_exists('A');
        detail = sprintf('loss_coarse=%.8g; time_LPCA=%.4g; vio_ratio=%.8g; A_bytes=%d', ...
            loss_coarse, time_LPCA, vio_ratio, A_bytes);
        records = append_record(records, "after_lpca", cfg, runtime, loss_coarse, NaN, A_bytes, detail);
    end
catch ME
    status = "failed";
    error_message = string(getReport(ME, 'extended', 'hyperlinks', 'off'));
    warning('%s', error_message);
    records = append_record(records, "error", cfg, NaN, NaN, NaN, NaN, char(ME.message));
end

finished_at = datetime("now");
out_base = sprintf('memory_probe_%s_%d_u%d_r%d_eps%s', ...
    cfg.city, cfg.node_count, cfg.user_id, cfg.repeat_id, strrep(num2str(cfg.epsilon), '.', 'p'));
csv_file = fullfile(result_dir, out_base + ".csv");
mat_file = fullfile(result_dir, out_base + ".mat");
writetable(records, csv_file);
save(mat_file, 'records', 'cfg', 'status', 'error_message', 'started_at', 'finished_at');

fprintf('Status: %s\n', status);
if status ~= "ok"
    fprintf('Error: %s\n', error_message);
end
fprintf('Wrote %s\n', csv_file);
fprintf('Wrote %s\n', mat_file);

function records = append_record(records, stage, cfg, runtime_sec, value1, value2, method_bytes, detail)
    [user_mem, sys_mem] = memory;
    row = table( ...
        datetime("now"), string(stage), string(cfg.city), cfg.node_count, cfg.user_id, cfg.repeat_id, cfg.epsilon, ...
        runtime_sec, value1, value2, method_bytes, ...
        user_mem.MemUsedMATLAB, user_mem.MemAvailableAllArrays, user_mem.MaxPossibleArrayBytes, ...
        sys_mem.PhysicalMemory.Available, sys_mem.PhysicalMemory.Total, string(detail), ...
        'VariableNames', {'timestamp','stage','city','node_count','user_id','repeat_id','epsilon', ...
        'runtime_sec','value1','value2','method_bytes','matlab_mem_used_bytes', ...
        'matlab_mem_available_all_arrays_bytes','matlab_max_possible_array_bytes', ...
        'physical_mem_available_bytes','physical_mem_total_bytes','detail'});
    records = [records; row];
end

function nbytes = bytes_if_exists(var_name)
    s = evalin('caller', sprintf('whos(''%s'')', var_name));
    if isempty(s)
        nbytes = NaN;
    else
        nbytes = s.bytes;
    end
end

function nbytes = bytes_of_var(x)
    s = whos('x');
    nbytes = s.bytes;
end

function est = estimate_full_lp_matrix_bytes(n_record, n_obf)
    n_pairs = n_record * (n_record - 1) / 2;
    est = struct();
    est.NR_RECORD = n_record;
    est.NR_OBF = n_obf;
    est.A_rows = 2 * n_obf * n_pairs;
    est.A_cols = n_record * n_obf;
    est.A_bytes = est.A_rows * est.A_cols * 8;
end

function est = estimate_lpca_matrix_bytes(col_longitude, col_latitude, obf_ID)
    x_min = min(col_longitude);
    x_max = max(col_longitude);
    y_min = min(col_latitude);
    y_max = max(col_latitude);
    num_grid = 8;
    edges_x = linspace(x_min, x_max, num_grid + 1);
    edges_y = linspace(y_min, y_max, num_grid + 1);
    rep_indices = NaN(num_grid, num_grid);

    for i = 1:num_grid
        for j = 1:num_grid
            if i < num_grid
                idx_x = col_longitude >= edges_x(i) & col_longitude < edges_x(i + 1);
            else
                idx_x = col_longitude >= edges_x(i) & col_longitude <= edges_x(i + 1);
            end
            if j < num_grid
                idx_y = col_latitude >= edges_y(j) & col_latitude < edges_y(j + 1);
            else
                idx_y = col_latitude >= edges_y(j) & col_latitude <= edges_y(j + 1);
            end
            idx = find(idx_x & idx_y);
            if ~isempty(idx)
                rep_indices(i, j) = idx(1);
            end
        end
    end

    n_record = nnz(~isnan(rep_indices));
    n_obf = numel(obf_ID);
    est = estimate_full_lp_matrix_bytes(n_record, n_obf);
end

function cfg = apply_env_override(cfg, field_name, env_name, is_string)
    value = getenv(env_name);
    if strlength(string(value)) == 0
        return;
    end
    if is_string
        cfg.(field_name) = char(value);
    else
        numeric_value = str2double(value);
        if isnan(numeric_value)
            error('Environment variable %s must be numeric, got: %s', env_name, value);
        end
        cfg.(field_name) = numeric_value;
    end
end

function cfg = apply_env_bool_override(cfg, field_name, env_name)
    value = lower(string(getenv(env_name)));
    if strlength(value) == 0
        return;
    end
    if any(value == ["1", "true", "yes", "y"])
        cfg.(field_name) = true;
    elseif any(value == ["0", "false", "no", "n"])
        cfg.(field_name) = false;
    else
        error('Environment variable %s must be boolean-like, got: %s', env_name, value);
    end
end
