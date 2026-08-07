% Memory-cost probe for the PAnDA baseline.
% Run from MATLAB with:
%   run('memory_cost_probe_outputs/panda_memory_probe.m')

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

cfg = apply_env_override(cfg, 'city', 'MEMPROBE_CITY', true);
cfg = apply_env_override(cfg, 'node_count', 'MEMPROBE_NODE_COUNT', false);
cfg = apply_env_override(cfg, 'user_id', 'MEMPROBE_USER_ID', false);
cfg = apply_env_override(cfg, 'repeat_id', 'MEMPROBE_REPEAT_ID', false);

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
    fprintf('PAnDA memory probe: %s, %d nodes, user %d, repeat %d\n', ...
        cfg.city, cfg.node_count, cfg.user_id, cfg.repeat_id);

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
    opts = detectImportOptions(node_csv);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_csv, opts);
    df_edges = readtable(edge_csv);
    col_longitude = table2array(df_nodes(:, 'x'));
    col_latitude = table2array(df_nodes(:, 'y'));

    parameters;
    env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
    env_parameters.NR_OBFLOC = numel(obf_ID);

    load(fullfile(base_dir, sprintf('G_%s.mat', cfg.city)), 'G');
    load_if_exists(fullfile(base_dir, sprintf('u_%s.mat', cfg.city)), 'u');
    load_if_exists(fullfile(base_dir, sprintf('v_%s.mat', cfg.city)), 'v');

    records = append_record(records, "after_load", cfg, NaN, NaN, NaN, "loaded inputs");

    mem_before = memory;
    t0 = tic;
    PAnDA;
    runtime_sec = toc(t0);
    mem_after = memory;

    panda_vars_bytes = selected_vars_bytes({'agent_2PPO','agent','masteragent','obf_matrix','obf_matrix_LB', ...
        'distance_matrix','distance_matrix_original','epsilon_nmw','xi_hathat','Pr','w','B_xn_xnhat'});
    detail = sprintf(['mem_delta_bytes=%d; selected_workspace_bytes=%d; ', ...
        'loss_ep4=%.8g; loss_ep7=%.8g; loss_ep10=%.8g; ', ...
        'time_2PPO_ep4=%.4g; time_2PPO_ep7=%.4g; time_2PPO_ep10=%.4g'], ...
        mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB, panda_vars_bytes, ...
        loss_ep4, loss_ep7, loss_ep10, time_2PPO_ep4, time_2PPO_ep7, time_2PPO_ep10);
    records = append_record(records, "after_panda", cfg, runtime_sec, ...
        mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB, panda_vars_bytes, detail);
catch ME
    status = "failed";
    error_message = string(getReport(ME, 'extended', 'hyperlinks', 'off'));
    warning('%s', error_message);
    records = append_record(records, "error", cfg, NaN, NaN, NaN, char(ME.message));
end

finished_at = datetime("now");
out_base = sprintf('panda_memory_probe_%s_%d_u%d_r%d', ...
    cfg.city, cfg.node_count, cfg.user_id, cfg.repeat_id);
csv_file = fullfile(result_dir, out_base + ".csv");
mat_file = fullfile(result_dir, out_base + ".mat");
writetable(records, csv_file);
save(mat_file, 'records', 'cfg', 'status', 'error_message', 'started_at', 'finished_at');

fprintf('Status: %s\n', status);
fprintf('Wrote %s\n', csv_file);
fprintf('Wrote %s\n', mat_file);

function records = append_record(records, stage, cfg, runtime_sec, value1, value2, detail)
    [user_mem, sys_mem] = memory;
    row = table( ...
        datetime("now"), string(stage), string(cfg.city), cfg.node_count, cfg.user_id, cfg.repeat_id, ...
        runtime_sec, value1, value2, user_mem.MemUsedMATLAB, ...
        user_mem.MemAvailableAllArrays, user_mem.MaxPossibleArrayBytes, ...
        sys_mem.PhysicalMemory.Available, sys_mem.PhysicalMemory.Total, string(detail), ...
        'VariableNames', {'timestamp','stage','city','node_count','user_id','repeat_id', ...
        'runtime_sec','value1','value2','matlab_mem_used_bytes', ...
        'matlab_mem_available_all_arrays_bytes','matlab_max_possible_array_bytes', ...
        'physical_mem_available_bytes','physical_mem_total_bytes','detail'});
    records = [records; row];
end

function total_bytes = selected_vars_bytes(var_names)
    total_bytes = 0;
    for i = 1:numel(var_names)
        info = evalin('caller', sprintf('whos(''%s'')', var_names{i}));
        if ~isempty(info)
            total_bytes = total_bytes + info.bytes;
        end
    end
end

function load_if_exists(file_name, variable_name)
    if isfile(file_name)
        data = load(file_name, variable_name);
        assignin('caller', variable_name, data.(variable_name));
    end
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
