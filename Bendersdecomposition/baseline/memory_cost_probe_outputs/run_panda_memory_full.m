% Full PAnDA memory-cost runner for the Table-IV-style baseline table.
%
% Defaults cover the full baseline setting:
%   cities: rome,london,nyc
%   sizes: 2000,4000,6000
%   users: 1:10
%   repeats: auto (london=1:6, rome/nyc=1:4)
%
% The runner is resumable. Existing ok rows in the output CSV are skipped.

clear;
clc;
rng(1);

script_dir = fileparts(mfilename('fullpath'));
cd(fullfile(script_dir, '..'));
base_dir = '.';
result_dir = fullfile('memory_cost_probe_outputs');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

cfg = struct();
cfg.cities = parse_list(getenv_default('PANDA_CITIES', 'rome,london,nyc'));
cfg.sizes = parse_numeric_list(getenv_default('PANDA_SIZES', '2000,4000,6000'));
cfg.users = parse_range(getenv_default('PANDA_USERS', '1:10'));
cfg.repeats_spec = getenv_default('PANDA_REPEATS', 'auto');
cfg.epsilons = [4 7 10];
cfg.output_tag = getenv_default('PANDA_OUTPUT_TAG', 'full');
cfg.output_csv = fullfile(result_dir, "panda_memory_" + cfg.output_tag + "_raw.csv");
cfg.output_mat = fullfile(result_dir, "panda_memory_" + cfg.output_tag + "_raw.mat");

addpath(fullfile(base_dir, 'functions'));
addpath(fullfile(base_dir, 'functions', 'haversine'));
addpath(fullfile(base_dir, 'functions', 'myBDToolbox'));
addpath(fullfile(base_dir, 'functions', 'myPlotToolbox'));
addpath(fullfile(base_dir, 'functions', 'myRLToolbox'));

if isfile(cfg.output_csv)
    raw = readtable(cfg.output_csv, 'TextType', 'string');
else
    raw = table();
end

fprintf('PAnDA full memory run started at %s\n', string(datetime("now")));
fprintf('Output: %s\n', cfg.output_csv);

for city_idx = 1:numel(cfg.cities)
    city = char(cfg.cities(city_idx));
    city_data = load_city_context(base_dir, city);
    repeats = resolve_repeats(cfg.repeats_spec, city);

    for size_idx = 1:numel(cfg.sizes)
        node_count = cfg.sizes(size_idx);

        for user_id = cfg.users
            for repeat_id = repeats
                if has_ok_case(raw, city, node_count, user_id, repeat_id)
                    fprintf('Skipping existing ok case: %s %d u%d r%d\n', city, node_count, user_id, repeat_id);
                    continue;
                end

                data_file = fullfile(base_dir, sprintf('%s_location_data_%d_nodes', city, node_count), ...
                    sprintf('location_data_sample_%d', user_id), ...
                    sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
                if ~isfile(data_file)
                    raw = append_case(raw, city, node_count, user_id, repeat_id, "missing", NaN, NaN, "data file not found");
                    save_progress(raw, cfg.output_csv, cfg.output_mat);
                    continue;
                end

                fprintf('Running PAnDA: %s, %d records, user %d, repeat %d at %s\n', ...
                    city, node_count, user_id, repeat_id, string(datetime("now")));
                try
                    S = load(data_file);
                    node_tar = S.node_tar;
                    obf_ID = S.obf_ID;
                    if isfield(S, 'lon_sel'); lon_sel = S.lon_sel; end %#ok<NASGU>
                    if isfield(S, 'lat_sel'); lat_sel = S.lat_sel; end %#ok<NASGU>

                    parameters;
                    env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
                    env_parameters.NR_OBFLOC = numel(obf_ID);
                    col_longitude = city_data.lon; %#ok<NASGU>
                    col_latitude = city_data.lat; %#ok<NASGU>
                    df_nodes = city_data.df_nodes; %#ok<NASGU>
                    df_edges = city_data.df_edges; %#ok<NASGU>
                    G = city_data.G; %#ok<NASGU>
                    if isfield(city_data, 'u'); u = city_data.u; end %#ok<NASGU>
                    if isfield(city_data, 'v'); v = city_data.v; end %#ok<NASGU>

                    mem_before = memory;
                    t0 = tic;
                    evalc('PAnDA');
                    runtime_sec = toc(t0);
                    mem_after = memory;

                    mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
                    selected_bytes = bytes_of_existing_vars({'agent_2PPO','agent','masteragent','obf_matrix','obf_matrix_LB', ...
                        'distance_matrix','distance_matrix_original','epsilon_nmw','xi_hathat','Pr','w','B_xn_xnhat'});
                    detail = sprintf('runtime=%.6g; mem_delta=%d; selected_workspace_bytes=%d', ...
                        runtime_sec, mem_delta, selected_bytes);
                    raw = append_case(raw, city, node_count, user_id, repeat_id, "ok", mem_delta, runtime_sec, detail);
                catch ME
                    raw = append_case(raw, city, node_count, user_id, repeat_id, "failed", NaN, NaN, ME.message);
                end

                save_progress(raw, cfg.output_csv, cfg.output_mat);
            end
        end
    end
end

summary = summarize_panda(raw, cfg.epsilons);
summary_csv = fullfile(result_dir, "panda_memory_" + cfg.output_tag + "_summary.csv");
writetable(summary, summary_csv);
fprintf('PAnDA full memory run finished at %s\n', string(datetime("now")));
fprintf('Wrote summary: %s\n', summary_csv);

function raw = append_case(raw, city, node_count, user_id, repeat_id, status, memory_cost_bytes, runtime_sec, detail)
    row = table(string(city), node_count, user_id, repeat_id, string(status), memory_cost_bytes, runtime_sec, string(detail), ...
        'VariableNames', {'city','node_count','user_id','repeat_id','status','memory_cost_bytes','runtime_sec','detail'});
    raw = [raw; row];
end

function tf = has_ok_case(raw, city, node_count, user_id, repeat_id)
    if isempty(raw)
        tf = false;
        return;
    end
    tf = any(raw.city == string(city) & raw.node_count == node_count & raw.user_id == user_id & ...
        raw.repeat_id == repeat_id & raw.status == "ok");
end

function save_progress(raw, csv_file, mat_file)
    writetable(raw, csv_file);
    save(mat_file, 'raw');
end

function summary = summarize_panda(raw, epsilons)
    ok = raw(raw.status == "ok", :);
    cities = unique(ok.city, 'stable');
    sizes = [2000 4000 6000];
    summary = table();
    for c = 1:numel(cities)
        vals = strings(1, numel(sizes) * numel(epsilons));
        idx = 1;
        for s = 1:numel(sizes)
            subset = ok(ok.city == cities(c) & ok.node_count == sizes(s), :);
            if isempty(subset)
                formatted = "---";
            else
                x = subset.memory_cost_bytes(~isnan(subset.memory_cost_bytes));
                mu = mean(x);
                if numel(x) > 1
                    ci = 1.96 * std(x) / sqrt(numel(x));
                else
                    ci = 0;
                end
                if ci == 0
                    formatted = format_bytes(mu);
                else
                    formatted = sprintf('%s±%s', format_bytes(mu), format_bytes(ci));
                end
            end
            for e = 1:numel(epsilons)
                vals(idx) = formatted;
                idx = idx + 1;
            end
        end
        row = table(cities(c), vals(1), vals(2), vals(3), vals(4), vals(5), vals(6), vals(7), vals(8), vals(9), ...
            'VariableNames', {'city','records2000_eps4','records2000_eps7','records2000_eps10', ...
            'records4000_eps4','records4000_eps7','records4000_eps10', ...
            'records6000_eps4','records6000_eps7','records6000_eps10'});
        summary = [summary; row];
    end
end

function city_data = load_city_context(base_dir, city)
    raw_dir = fullfile(base_dir, 'Dataset', city, 'raw');
    node_csv = fullfile(raw_dir, sprintf('%s_nodes.csv', city));
    edge_csv = fullfile(raw_dir, sprintf('%s_edges.csv', city));
    opts = detectImportOptions(node_csv);
    opts = setvartype(opts, 'osmid', 'int64');
    city_data = struct();
    city_data.df_nodes = readtable(node_csv, opts);
    city_data.df_edges = readtable(edge_csv);
    city_data.lon = table2array(city_data.df_nodes(:, 'x'));
    city_data.lat = table2array(city_data.df_nodes(:, 'y'));
    loaded = load(fullfile(base_dir, sprintf('G_%s.mat', city)), 'G');
    city_data.G = loaded.G;
    u_file = fullfile(base_dir, sprintf('u_%s.mat', city));
    if isfile(u_file)
        loaded = load(u_file, 'u');
        city_data.u = loaded.u;
    end
    v_file = fullfile(base_dir, sprintf('v_%s.mat', city));
    if isfile(v_file)
        loaded = load(v_file, 'v');
        city_data.v = loaded.v;
    end
end

function total = bytes_of_existing_vars(names)
    total = 0;
    for i = 1:numel(names)
        info = evalin('caller', sprintf('whos(''%s'')', names{i}));
        if ~isempty(info)
            total = total + info.bytes;
        end
    end
end

function out = format_bytes(bytes)
    if isnan(bytes)
        out = "---";
    elseif bytes >= 1e12
        out = sprintf('%.1f TB', bytes / 1e12);
    elseif bytes >= 1e9
        out = sprintf('%.2f GB', bytes / 1e9);
    elseif bytes >= 1e6
        out = sprintf('%.2f MB', bytes / 1e6);
    elseif bytes >= 1e3
        out = sprintf('%.2f KB', bytes / 1e3);
    else
        out = sprintf('%.0f B', bytes);
    end
end

function repeats = resolve_repeats(spec, city)
    if strcmpi(spec, 'auto')
        if strcmpi(city, 'london')
            repeats = 1:6;
        else
            repeats = 1:4;
        end
    else
        repeats = parse_range(spec);
    end
end

function value = getenv_default(name, default_value)
    value = getenv(name);
    if strlength(string(value)) == 0
        value = default_value;
    end
end

function values = parse_list(text)
    parts = split(string(text), ',');
    values = strip(parts(parts ~= ""));
end

function values = parse_numeric_list(text)
    parts = parse_list(text);
    values = zeros(1, numel(parts));
    for i = 1:numel(parts)
        values(i) = str2double(parts(i));
    end
end

function values = parse_range(text)
    s = string(text);
    if contains(s, ':')
        parts = split(s, ':');
        values = str2double(parts(1)):str2double(parts(2));
    else
        values = parse_numeric_list(s);
    end
end
