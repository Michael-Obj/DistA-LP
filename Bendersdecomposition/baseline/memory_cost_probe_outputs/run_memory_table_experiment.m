% Table-IV-style memory-cost experiment for baseline methods.
%
% Default mode runs one sanity sample:
%   city=london, records=2000, user=1, repeat=1
%
% Full run example from PowerShell:
%   $env:MEMTABLE_CITIES='rome,london,nyc'
%   $env:MEMTABLE_SIZES='2000,4000,6000'
%   $env:MEMTABLE_USERS='1:10'
%   $env:MEMTABLE_REPEATS='auto'
%   matlab -batch "run('memory_cost_probe_outputs/run_memory_table_experiment.m')"
%
% Optional:
%   MEMTABLE_RUN_PANDA=true/false     default true
%   MEMTABLE_SOLVE_LPA=true/false     default false; false records LP-A A-matrix memory only

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
cfg.cities = parse_list(getenv_default('MEMTABLE_CITIES', 'london'));
cfg.sizes = parse_numeric_list(getenv_default('MEMTABLE_SIZES', '2000'));
cfg.users = parse_range(getenv_default('MEMTABLE_USERS', '1'));
cfg.repeats_spec = getenv_default('MEMTABLE_REPEATS', '1');
cfg.epsilons = parse_numeric_list(getenv_default('MEMTABLE_EPSILONS', '4,7,10'));
cfg.run_panda = parse_bool(getenv_default('MEMTABLE_RUN_PANDA', 'true'));
cfg.solve_lpa = parse_bool(getenv_default('MEMTABLE_SOLVE_LPA', 'false'));

addpath(fullfile(base_dir, 'functions'));
addpath(fullfile(base_dir, 'functions', 'haversine'));
addpath(fullfile(base_dir, 'functions', 'myBDToolbox'));
addpath(fullfile(base_dir, 'functions', 'myPlotToolbox'));
addpath(fullfile(base_dir, 'functions', 'myRLToolbox'));

raw = table();
fprintf('Memory table experiment started at %s\n', string(datetime("now")));

for city_idx = 1:numel(cfg.cities)
    city = char(cfg.cities(city_idx));
    city_data = load_city_context(base_dir, city);

    for size_idx = 1:numel(cfg.sizes)
        node_count = cfg.sizes(size_idx);
        repeats = resolve_repeats(cfg.repeats_spec, city);

        for user_id = cfg.users
            for repeat_id = repeats
                data_file = fullfile(base_dir, sprintf('%s_location_data_%d_nodes', city, node_count), ...
                    sprintf('location_data_sample_%d', user_id), ...
                    sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
                if ~isfile(data_file)
                    raw = append_raw(raw, city, node_count, user_id, repeat_id, NaN, "ALL", ...
                        "missing", NaN, NaN, "data file not found");
                    continue;
                end

                fprintf('Case: %s, %d records, user %d, repeat %d\n', city, node_count, user_id, repeat_id);
                S = load(data_file);
                node_tar = S.node_tar;
                obf_ID = S.obf_ID;
                if isfield(S, 'lon_sel'); lon_sel = S.lon_sel; else; lon_sel = city_data.lon(node_tar); end
                if isfield(S, 'lat_sel'); lat_sel = S.lat_sel; else; lat_sel = city_data.lat(node_tar); end

                parameters;
                env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
                env_parameters.NR_OBFLOC = numel(obf_ID);

                distance_matrix = distanceMatrix(city_data.lon(node_tar), city_data.lat(node_tar));
                task_loc = 2;
                cost_matrix = build_cost_matrix(node_tar, obf_ID, city_data.G, task_loc);

                lp_est = estimate_full_lp_matrix_bytes(numel(node_tar), numel(obf_ID));
                lpa_est = estimate_lpa_matrix_bytes(city_data.lon, city_data.lat, numel(obf_ID));

                for eps_idx = 1:numel(cfg.epsilons)
                    epsilon = cfg.epsilons(eps_idx);
                    env_parameters.EPSILON = epsilon;

                    raw = record_em(raw, city, node_count, user_id, repeat_id, epsilon, env_parameters, ...
                        obf_ID, distance_matrix, node_tar, city_data.G, task_loc, cost_matrix);
                    raw = record_laplace(raw, city, node_count, user_id, repeat_id, epsilon, ...
                        lon_sel, lat_sel, city_data.lon(obf_ID'), city_data.lat(obf_ID'), cost_matrix);
                    raw = record_rmp(raw, city, node_count, user_id, repeat_id, epsilon, env_parameters, ...
                        obf_ID, distance_matrix, cost_matrix);

                    detail = sprintf('Dense full-LP A estimate only: rows=%g; cols=%g; bytes=%g', ...
                        lp_est.A_rows, lp_est.A_cols, lp_est.A_bytes);
                    raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "LP", ...
                        "estimated_infeasible", lp_est.A_bytes, NaN, detail);

                    if cfg.solve_lpa
                        raw = record_lpa_solve(raw, city, node_count, user_id, repeat_id, epsilon, ...
                            env_parameters, city_data, node_tar, obf_ID, distance_matrix);
                    else
                        detail = sprintf('LP-A coarse dense A estimate only: coarse_records=%d; obf=%d; rows=%g; cols=%g; bytes=%g', ...
                            lpa_est.NR_RECORD, lpa_est.NR_OBF, lpa_est.A_rows, lpa_est.A_cols, lpa_est.A_bytes);
                        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "LP-A", ...
                            "estimated", lpa_est.A_bytes, NaN, detail);
                    end

                    raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "COPT", ...
                        "not_available", NaN, NaN, "COPT implementation is not present in the provided baseline folder");
                end

                if cfg.run_panda
                    panda_rows = record_panda(raw, city, node_count, user_id, repeat_id, cfg.epsilons, ...
                        env_parameters, city_data, node_tar, obf_ID, S);
                    raw = panda_rows;
                else
                    for eps_idx = 1:numel(cfg.epsilons)
                        raw = append_raw(raw, city, node_count, user_id, repeat_id, cfg.epsilons(eps_idx), "PAnDA", ...
                            "skipped", NaN, NaN, "MEMTABLE_RUN_PANDA=false");
                    end
                end

                raw_file_live = fullfile(result_dir, 'memory_table_raw_live.csv');
                writetable(raw, raw_file_live);
            end
        end
    end
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
raw_file = fullfile(result_dir, "memory_table_raw_" + timestamp + ".csv");
summary_file = fullfile(result_dir, "memory_table_summary_" + timestamp + ".csv");
tex_file = fullfile(result_dir, "memory_table_summary_" + timestamp + ".tex");
writetable(raw, raw_file);
summary = summarize_raw(raw);
writetable(summary, summary_file);
write_latex_summary(summary, tex_file);

fprintf('Wrote raw results: %s\n', raw_file);
fprintf('Wrote summary CSV: %s\n', summary_file);
fprintf('Wrote summary LaTeX: %s\n', tex_file);

function raw = record_em(raw, city, node_count, user_id, repeat_id, epsilon, env_parameters, obf_ID, distance_matrix, node_tar, G, task_loc, cost_matrix)
    try
        mem_before = memory;
        t0 = tic;
        P_matrix = zeros(length(distance_matrix), length(obf_ID));
        sum_i = zeros(length(distance_matrix), 1);
        for i = 1:length(distance_matrix)
            for j = 1:length(obf_ID)
                sum_i(i,1) = sum_i(i,1) + exp(-epsilon * distance_matrix(i, obf_ID(j)) / 2.0);
            end
            for j = 1:length(obf_ID)
                P_matrix(i,j) = exp(-epsilon * distance_matrix(i, obf_ID(j)) / 2.0) / sum_i(i,1);
            end
        end
        loss_em = sum(sum(cost_matrix .* P_matrix));
        runtime = toc(t0);
        mem_after = memory;
        major_bytes = bytes_of_vars({'cost_matrix','P_matrix','sum_i'});
        mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
        detail = sprintf('loss=%.8g; mem_delta=%d; major_bytes=%d', loss_em, mem_delta, major_bytes);
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "EM", "ok", major_bytes, runtime, detail);
    catch ME
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "EM", "failed", NaN, NaN, ME.message);
    end
end

function raw = record_laplace(raw, city, node_count, user_id, repeat_id, epsilon, loc_lons, loc_lats, pert_lons, pert_lats, cost_matrix)
    try
        mem_before = memory;
        t0 = tic;
        [K, QL, time_laplace] = planar_laplace_utility_loss(loc_lons, loc_lats, pert_lons, pert_lats, cost_matrix, epsilon);
        runtime = toc(t0);
        mem_after = memory;
        major_bytes = bytes_of_vars({'cost_matrix','K'});
        mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
        detail = sprintf('loss=%.8g; time_laplace=%.4g; mem_delta=%d; major_bytes=%d', QL, time_laplace, mem_delta, major_bytes);
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "Laplace", "ok", major_bytes, runtime, detail);
    catch ME
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "Laplace", "failed", NaN, NaN, ME.message);
    end
end

function raw = record_rmp(raw, city, node_count, user_id, repeat_id, epsilon, env_parameters, obf_ID, distance_matrix, cost_matrix)
    try
        mem_before = memory;
        t0 = tic;
        P_matrix = zeros(length(distance_matrix), length(obf_ID));
        sum_i = zeros(length(distance_matrix), 1);
        for i = 1:length(distance_matrix)
            for j = 1:length(obf_ID)
                sum_i(i,1) = sum_i(i,1) + exp(-epsilon * distance_matrix(i, obf_ID(j)) / 2.0);
            end
            for j = 1:length(obf_ID)
                P_matrix(i,j) = exp(-epsilon * distance_matrix(i, obf_ID(j)) / 2.0) / sum_i(i,1);
            end
        end
        P_2 = zeros(length(obf_ID), length(distance_matrix));
        for i = 1:length(obf_ID)
            for j = 1:length(distance_matrix)
                P_2(i,j) = P_matrix(j,i) / sum(P_matrix(:,i));
            end
        end
        y_k = sparse(length(distance_matrix), 0);
        for i = 1:length(obf_ID)
            sum_pc = [];
            for j = 1:length(obf_ID)
                sum_pc_j = P_2(i,:) * cost_matrix(:,j);
                sum_pc = [sum_pc, sum_pc_j];
            end
            [~, y_k(i)] = min(sum_pc);
        end
        rmp_loss = sum(sum(cost_matrix(:,y_k) .* P_matrix));
        runtime = toc(t0);
        mem_after = memory;
        major_bytes = bytes_of_vars({'cost_matrix','P_matrix','P_2','y_k','sum_i'});
        mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
        detail = sprintf('loss=%.8g; mem_delta=%d; major_bytes=%d', rmp_loss, mem_delta, major_bytes);
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "RMP", "ok", major_bytes, runtime, detail);
    catch ME
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "RMP", "failed", NaN, NaN, ME.message);
    end
end

function raw = record_lpa_solve(raw, city, node_count, user_id, repeat_id, epsilon, env_parameters, city_data, node_tar, obf_ID, distance_matrix)
    try
        col_longitude = city_data.lon;
        col_latitude = city_data.lat;
        df_nodes = city_data.df_nodes;
        df_edges = city_data.df_edges;
        G = city_data.G;
        distance_save = distance_matrix;
        env_parameters.EPSILON = epsilon;
        mem_before = memory;
        t0 = tic;
        evalc('coarse');
        runtime = toc(t0);
        mem_after = memory;
        A_info = whos('A');
        if isempty(A_info); major_bytes = NaN; else; major_bytes = A_info.bytes; end
        mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
        detail = sprintf('loss=%.8g; time_LPCA=%.4g; mem_delta=%d; A_bytes=%g', loss_coarse, time_LPCA, mem_delta, major_bytes);
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "LP-A", "ok", major_bytes, runtime, detail);
    catch ME
        raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, "LP-A", "failed", NaN, NaN, ME.message);
    end
end

function raw = record_panda(raw, city, node_count, user_id, repeat_id, epsilons, env_parameters, city_data, node_tar, obf_ID, S)
    try
        col_longitude = city_data.lon;
        col_latitude = city_data.lat;
        df_nodes = city_data.df_nodes;
        df_edges = city_data.df_edges;
        G = city_data.G;
        if isfield(city_data, 'u'); u = city_data.u; end
        if isfield(city_data, 'v'); v = city_data.v; end
        if isfield(S, 'lon_sel'); lon_sel = S.lon_sel; end
        if isfield(S, 'lat_sel'); lat_sel = S.lat_sel; end

        mem_before = memory;
        t0 = tic;
        evalc('PAnDA');
        runtime = toc(t0);
        mem_after = memory;
        mem_delta = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
        selected_bytes = bytes_of_existing_vars({'agent_2PPO','agent','masteragent','obf_matrix','obf_matrix_LB', ...
            'distance_matrix','distance_matrix_original','epsilon_nmw','xi_hathat','Pr','w','B_xn_xnhat'});
        detail = sprintf('PAnDA runs epsilon 4/7/10 in one script; runtime=%.4g; mem_delta=%d; selected_bytes=%d', ...
            runtime, mem_delta, selected_bytes);
        for eps_idx = 1:numel(epsilons)
            raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilons(eps_idx), "PAnDA", "ok", mem_delta, runtime, detail);
        end
    catch ME
        for eps_idx = 1:numel(epsilons)
            raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilons(eps_idx), "PAnDA", "failed", NaN, NaN, ME.message);
        end
    end
end

function cost_matrix = build_cost_matrix(node_tar, obf_ID, G, task_loc)
    nr_loc = length(node_tar);
    nr_obf = length(obf_ID);
    cost_matrix = zeros(nr_loc, nr_obf);
    [~, D] = shortestpathtree(G, node_tar(task_loc));
    for i = 1:nr_loc
        for j = 1:nr_obf
            cost_matrix(i,j) = abs(D(node_tar(i)) - D(node_tar(obf_ID(j))));
        end
    end
    cost_matrix = cost_matrix / nr_loc;
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
    g_file = fullfile(base_dir, sprintf('G_%s.mat', city));
    loaded = load(g_file, 'G');
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

function raw = append_raw(raw, city, node_count, user_id, repeat_id, epsilon, method, status, memory_cost_bytes, runtime_sec, detail)
    row = table(string(city), node_count, user_id, repeat_id, epsilon, string(method), string(status), ...
        memory_cost_bytes, runtime_sec, string(detail), ...
        'VariableNames', {'city','node_count','user_id','repeat_id','epsilon','method','status', ...
        'memory_cost_bytes','runtime_sec','detail'});
    raw = [raw; row];
end

function summary = summarize_raw(raw)
    ok = raw(raw.status == "ok" | raw.status == "estimated" | raw.status == "estimated_infeasible" | raw.status == "not_available", :);
    cities = unique(ok.city, 'stable');
    methods = ["EM","Laplace","RMP","COPT","LP","LP-A","PAnDA"];
    sizes = [2000 4000 6000];
    epsilons = [4 7 10];
    summary = table();
    for c = 1:numel(cities)
        for m = 1:numel(methods)
            vals = strings(1, numel(sizes) * numel(epsilons));
            idx = 1;
            for s = 1:numel(sizes)
                for e = 1:numel(epsilons)
                    subset = ok(ok.city == cities(c) & ok.method == methods(m) & ok.node_count == sizes(s) & ok.epsilon == epsilons(e), :);
                    if isempty(subset) || all(isnan(subset.memory_cost_bytes))
                        vals(idx) = "---";
                    else
                        x = subset.memory_cost_bytes(~isnan(subset.memory_cost_bytes));
                        mu = mean(x);
                        if numel(x) > 1
                            ci = 1.96 * std(x) / sqrt(numel(x));
                        else
                            ci = 0;
                        end
                        if ci == 0
                            vals(idx) = format_bytes(mu);
                        else
                            vals(idx) = sprintf('%s±%s', format_bytes(mu), format_bytes(ci));
                        end
                    end
                    idx = idx + 1;
                end
            end
            row = table(cities(c), methods(m), vals(1), vals(2), vals(3), vals(4), vals(5), vals(6), vals(7), vals(8), vals(9), ...
                'VariableNames', {'city','method','records2000_eps4','records2000_eps7','records2000_eps10', ...
                'records4000_eps4','records4000_eps7','records4000_eps10', ...
                'records6000_eps4','records6000_eps7','records6000_eps10'});
            summary = [summary; row];
        end
    end
end

function write_latex_summary(summary, tex_file)
    fid = fopen(tex_file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '\\begin{table*}[t]\n\\centering\n');
    fprintf(fid, '\\caption{Memory cost across different baseline perturbation methods. Mean$\\pm$1.96$\\times$standard error; unavailable results are labeled by ``---''.}\n');
    fprintf(fid, '\\resizebox{\\textwidth}{!}{%%\n');
    fprintf(fid, '\\begin{tabular}{ll|ccc|ccc|ccc}\n\\hline\n');
    fprintf(fid, '\\multicolumn{2}{c|}{Number of records} & \\multicolumn{3}{c|}{2,000} & \\multicolumn{3}{c|}{4,000} & \\multicolumn{3}{c}{6,000} \\\\\n');
    fprintf(fid, '\\multicolumn{2}{c|}{Privacy budget (km$^{-1}$)} & $\\epsilon=4.0$ & $\\epsilon=7.0$ & $\\epsilon=10.0$ & $\\epsilon=4.0$ & $\\epsilon=7.0$ & $\\epsilon=10.0$ & $\\epsilon=4.0$ & $\\epsilon=7.0$ & $\\epsilon=10.0$ \\\\\n\\hline\n');
    cities = unique(summary.city, 'stable');
    for c = 1:numel(cities)
        fprintf(fid, '\\multicolumn{11}{c}{%s road map} \\\\\n\\hline\n', city_title(cities(c)));
        subset = summary(summary.city == cities(c), :);
        for i = 1:height(subset)
            group = method_group(subset.method(i));
            fprintf(fid, '%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n', ...
                group, subset.method(i), subset.records2000_eps4(i), subset.records2000_eps7(i), subset.records2000_eps10(i), ...
                subset.records4000_eps4(i), subset.records4000_eps7(i), subset.records4000_eps10(i), ...
                subset.records6000_eps4(i), subset.records6000_eps7(i), subset.records6000_eps10(i));
        end
        fprintf(fid, '\\hline\n');
    end
    fprintf(fid, '\\end{tabular}%%\n}\n\\end{table*}\n');
end

function title = city_title(city)
    if city == "nyc"
        title = "New York City";
    elseif city == "rome"
        title = "Rome";
    elseif city == "london"
        title = "London";
    else
        title = char(city);
    end
end

function group = method_group(method)
    if method == "EM" || method == "Laplace"
        group = "Pre-defined Noise Distribution";
    elseif method == "RMP" || method == "COPT"
        group = "Hybrid Method";
    else
        group = "Optimization Based Methods";
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

function total = bytes_of_vars(names)
    total = 0;
    for i = 1:numel(names)
        info = evalin('caller', sprintf('whos(''%s'')', names{i}));
        if ~isempty(info)
            total = total + info.bytes;
        end
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

function est = estimate_full_lp_matrix_bytes(n_record, n_obf)
    n_pairs = n_record * (n_record - 1) / 2;
    est = struct();
    est.NR_RECORD = n_record;
    est.NR_OBF = n_obf;
    est.A_rows = 2 * n_obf * n_pairs;
    est.A_cols = n_record * n_obf;
    est.A_bytes = est.A_rows * est.A_cols * 8;
end

function est = estimate_lpa_matrix_bytes(lon, lat, n_obf)
    num_grid = 8;
    edges_x = linspace(min(lon), max(lon), num_grid + 1);
    edges_y = linspace(min(lat), max(lat), num_grid + 1);
    n_record = 0;
    for i = 1:num_grid
        for j = 1:num_grid
            if i < num_grid
                idx_x = lon >= edges_x(i) & lon < edges_x(i + 1);
            else
                idx_x = lon >= edges_x(i) & lon <= edges_x(i + 1);
            end
            if j < num_grid
                idx_y = lat >= edges_y(j) & lat < edges_y(j + 1);
            else
                idx_y = lat >= edges_y(j) & lat <= edges_y(j + 1);
            end
            if any(idx_x & idx_y)
                n_record = n_record + 1;
            end
        end
    end
    est = estimate_full_lp_matrix_bytes(n_record, n_obf);
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

function value = parse_bool(text)
    s = lower(string(text));
    value = any(s == ["1","true","yes","y"]);
end
