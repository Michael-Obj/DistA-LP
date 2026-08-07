% Estimate PAnDA memory from the selected point set size.
% This does not run Benders/LP. It reproduces PAnDA's point-selection stage
% and estimates the LP constraint matrix memory using n_selected and K.

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
cfg.cities = parse_list(getenv_default('PANDA_EST_CITIES', 'rome,london,nyc'));
cfg.sizes = parse_numeric_list(getenv_default('PANDA_EST_SIZES', '2000,4000,6000'));
cfg.users = parse_range(getenv_default('PANDA_EST_USERS', '1:10'));
cfg.repeats_spec = getenv_default('PANDA_EST_REPEATS', 'auto');
cfg.epsilons = [4 7 10];

addpath(fullfile(base_dir, 'functions'));
addpath(fullfile(base_dir, 'functions', 'haversine'));

raw = table();

for city_idx = 1:numel(cfg.cities)
    city = char(cfg.cities(city_idx));
    city_data = load_city_context(base_dir, city);
    repeats = resolve_repeats(cfg.repeats_spec, city);

    for size_idx = 1:numel(cfg.sizes)
        node_count = cfg.sizes(size_idx);
        for user_id = cfg.users
            for repeat_id = repeats
                data_file = fullfile(base_dir, sprintf('%s_location_data_%d_nodes', city, node_count), ...
                    sprintf('location_data_sample_%d', user_id), ...
                    sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
                if ~isfile(data_file)
                    continue;
                end

                fprintf('Estimating PAnDA: %s %d user %d repeat %d\n', city, node_count, user_id, repeat_id);
                S = load(data_file, 'node_tar', 'obf_ID');
                node_tar = S.node_tar;
                obf_ID = S.obf_ID;

                parameters;
                env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
                env_parameters.NR_OBFLOC = numel(obf_ID);

                distance_matrix = distanceMatrix(city_data.lon(node_tar), city_data.lat(node_tar));
                num_user = env_parameters.NR_AGENT;
                selected_user = randperm(env_parameters.NR_NODE_IN_TARGET, num_user);

                lambda = 0.5;
                alpha_hat = 0.95;
                D_MAX = max(max(distance_matrix));
                range_threshold = D_MAX / 150;
                w = getq(distance_matrix, lambda, range_threshold, alpha_hat);
                [~, all_target] = get_relevant_location_set(w, selected_user);

                n_selected = length(all_target);
                k_obf = length(obf_ID);
                dense = estimate_dense_lp(n_selected, k_obf);
                sparse = estimate_sparse_lp(n_selected, k_obf);
                selected_distance = distance_matrix(all_target, all_target);
                selected_adj = double(selected_distance <= env_parameters.NEIGHBOR_THRESHOLD);
                selected_adj(1:n_selected+1:end) = 0;
                benders = estimate_benders_sparse(selected_distance, selected_adj, min(25, n_selected), k_obf);
                selection_workspace_bytes = bytes_of_vars({'distance_matrix','w'});

                for eps_idx = 1:numel(cfg.epsilons)
                    row = table(string(city), node_count, user_id, repeat_id, cfg.epsilons(eps_idx), ...
                        n_selected, k_obf, dense.A_rows, dense.A_cols, dense.A_bytes, ...
                        sparse.nnz, sparse.estimated_bytes, benders.master_bytes, benders.subproblem_bytes, ...
                        benders.total_bytes, benders.boundary_nodes, benders.boundary_edges, benders.intra_edges, ...
                        selection_workspace_bytes, ...
                        'VariableNames', {'city','node_count','user_id','repeat_id','epsilon', ...
                        'n_selected','k_obf','dense_A_rows','dense_A_cols','dense_A_bytes', ...
                        'centralized_sparse_A_nnz','centralized_sparse_A_estimated_bytes', ...
                        'benders_master_estimated_bytes','benders_subproblem_estimated_bytes', ...
                        'benders_total_estimated_bytes','benders_boundary_nodes','benders_boundary_edges', ...
                        'benders_intra_edges','selection_workspace_bytes'});
                    raw = [raw; row];
                end
            end
        end
    end
end

raw_file = fullfile(result_dir, 'panda_selected_set_memory_estimates_raw.csv');
summary_file = fullfile(result_dir, 'panda_selected_set_memory_estimates_summary.csv');
writetable(raw, raw_file);
writetable(summarize(raw), summary_file);
fprintf('Wrote %s\n', raw_file);
fprintf('Wrote %s\n', summary_file);

function summary = summarize(raw)
    cities = unique(raw.city, 'stable');
    sizes = [2000 4000 6000];
    epsilons = [4 7 10];
    summary = table();
    for c = 1:numel(cities)
        vals = strings(1, 9);
        idx = 1;
        for s = 1:numel(sizes)
            subset = raw(raw.city == cities(c) & raw.node_count == sizes(s), :);
            x = subset.benders_total_estimated_bytes;
            if isempty(x)
                formatted = "---";
            else
                mu = mean(x);
                ci = 1.96 * std(x) / sqrt(numel(x));
                formatted = sprintf('%s±%s', format_bytes(mu), format_bytes(ci));
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

function est = estimate_dense_lp(n, k)
    rows = k * n * (n - 1);
    cols = n * k;
    est = struct('A_rows', rows, 'A_cols', cols, 'A_bytes', rows * cols * 8);
end

function est = estimate_sparse_lp(n, k)
    rows = k * n * (n - 1);
    cols = n * k;
    nnz_a = 2 * rows;
    estimated_bytes = 16 * nnz_a + 8 * (cols + 1);
    est = struct('nnz', nnz_a, 'estimated_bytes', estimated_bytes);
end

function est = estimate_benders_sparse(distance_matrix, adj, nr_agent, k)
    n = size(distance_matrix, 1);
    if n == 0
        est = struct('master_bytes', 0, 'subproblem_bytes', 0, 'total_bytes', 0, ...
            'boundary_nodes', 0, 'boundary_edges', 0, 'intra_edges', 0);
        return;
    end
    if nr_agent <= 1
        cluster_idx = ones(n, 1);
    else
        cluster_idx = kmeans(distance_matrix, nr_agent, 'Replicates', 1, 'MaxIter', 100, 'Display', 'off');
    end

    [row, col] = find(adj > 0);
    intra_edges = 0;
    boundary_edges = 0;
    boundary_node_mask = false(n, 1);
    subproblem_bytes = 0;

    for a = 1:nr_agent
        node_mask = cluster_idx == a;
        node_ids = find(node_mask);
        boundary_mask = false(n, 1);
        local_intra_edges = 0;

        for e = 1:numel(row)
            i = row(e);
            j = col(e);
            if cluster_idx(i) == a && cluster_idx(j) == a
                local_intra_edges = local_intra_edges + 1;
            elseif cluster_idx(i) == a && cluster_idx(j) ~= a
                boundary_mask(i) = true;
            end
        end

        internal_count = numel(setdiff(node_ids, find(boundary_mask)));
        boundary_count = nnz(boundary_mask);
        geo_rows = 2 * k * local_intra_edges;
        geo_cols = (internal_count + boundary_count) * k;
        geo_nnz = 4 * k * local_intra_edges;

        % subProblem builds A from A_dp plus two unit-sum blocks A_um.
        a_rows = geo_rows + 2 * internal_count;
        a_cols = internal_count * k;
        a_nnz = geo_nnz + 2 * internal_count * k;

        % Approximate sparse storage for agent.GeoI and subproblem A.
        subproblem_bytes = subproblem_bytes + sparse_bytes(geo_nnz, geo_cols) + sparse_bytes(a_nnz, a_cols);
        intra_edges = intra_edges + local_intra_edges;
    end

    for e = 1:numel(row)
        i = row(e);
        j = col(e);
        if cluster_idx(i) ~= cluster_idx(j)
            boundary_edges = boundary_edges + 1;
            boundary_node_mask(i) = true;
            boundary_node_mask(j) = true;
        end
    end

    boundary_nodes = nnz(boundary_node_mask);
    master_geo_rows = boundary_edges * k;
    master_geo_cols = boundary_nodes * k;
    master_geo_nnz = 2 * boundary_edges * k;
    master_aeq_nnz = boundary_nodes * k;
    master_cols_with_z = boundary_nodes * k + nr_agent;
    master_bytes = sparse_bytes(master_geo_nnz, master_geo_cols) + ...
        sparse_bytes(master_aeq_nnz, master_cols_with_z);

    est = struct('master_bytes', master_bytes, 'subproblem_bytes', subproblem_bytes, ...
        'total_bytes', master_bytes + subproblem_bytes, 'boundary_nodes', boundary_nodes, ...
        'boundary_edges', boundary_edges, 'intra_edges', intra_edges);
end

function bytes = sparse_bytes(nnz_count, n_cols)
    bytes = 16 * nnz_count + 8 * (n_cols + 1);
end

function city_data = load_city_context(base_dir, city)
    raw_dir = fullfile(base_dir, 'Dataset', city, 'raw');
    node_csv = fullfile(raw_dir, sprintf('%s_nodes.csv', city));
    opts = detectImportOptions(node_csv);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_csv, opts);
    city_data = struct();
    city_data.lon = table2array(df_nodes(:, 'x'));
    city_data.lat = table2array(df_nodes(:, 'y'));
end

function n = bytes_of_vars(names)
    n = 0;
    for i = 1:numel(names)
        info = evalin('caller', sprintf('whos(''%s'')', names{i}));
        if ~isempty(info)
            n = n + info.bytes;
        end
    end
end

function out = format_bytes(bytes)
    if bytes >= 1e12
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
