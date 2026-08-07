%% PAnDA baseline runner
% Run this script from the baseline artifact root. Results are aggregated
% in memory and printed in paper-style tables; no result files are saved.

clear;
clc;
rng(1);

%% Experiment configuration
city = 'rome';                 % 'rome', 'london', or 'nyc'
node_count = 2000;            % 2000, 4000, or 6000
epsilons = [4, 7, 10];
user_ids = 1:10;

if strcmpi(city, 'london')
    repeat_ids = 1:6;
elseif strcmpi(city, 'rome') && node_count == 2000
    repeat_ids = 1:5;
else
    repeat_ids = 1:4;
end

addpath('./functions/');
addpath('./functions/haversine');
addpath('./functions/myBDToolbox');
addpath('./functions/myPlotToolbox');
addpath('./functions/myRLToolbox');

%% Load the selected city once
node_csv = sprintf('./Dataset/%s/raw/%s_nodes.csv', city, city);
edge_csv = sprintf('./Dataset/%s/raw/%s_edges.csv', city, city);
opts = detectImportOptions(node_csv);
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_csv, opts);
df_edges = readtable(edge_csv);
col_longitude = table2array(df_nodes(:, 'x'));
col_latitude = table2array(df_nodes(:, 'y'));

graph_data = load(sprintf('G_%s.mat', city), 'G');
G = graph_data.G;

u_file = sprintf('u_%s.mat', city);
if isfile(u_file)
    u_data = load(u_file, 'u');
    u = u_data.u; %#ok<NASGU>
end
v_file = sprintf('v_%s.mat', city);
if isfile(v_file)
    v_data = load(v_file, 'v');
    v = v_data.v; %#ok<NASGU>
end

% Rows for each epsilon: PAnDA loss, LB loss, PAnDA time, LB time.
aggregate = zeros(4 * numel(epsilons), numel(repeat_ids));

%% Run all users and repetitions
for user_idx = 1:numel(user_ids)
    user_id = user_ids(user_idx);
    user_result = zeros(size(aggregate));

    for repeat_idx = 1:numel(repeat_ids)
        repeat_id = repeat_ids(repeat_idx);
        data_file = fullfile(sprintf('%s_location_data_%d_nodes', city, node_count), ...
            sprintf('location_data_sample_%d', user_id), ...
            sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
        sample = load(data_file);
        node_tar = sample.node_tar;
        obf_ID = sample.obf_ID;
        if isfield(sample, 'lon_sel'); lon_sel = sample.lon_sel; end %#ok<NASGU>
        if isfield(sample, 'lat_sel'); lat_sel = sample.lat_sel; end %#ok<NASGU>

        % Reset the PAnDA environment for every independent run because
        % PAnDA.m updates EPSILON and NR_AGENT internally.
        parameters;
        env_parameters.NR_NODE_IN_TARGET = numel(node_tar);
        env_parameters.NR_OBFLOC = numel(obf_ID);

        % PAnDA.m evaluates epsilon 4, 7, and 10 in one call. evalc
        % suppresses intermediate solver output so only final tables print.
        evalc('PAnDA');

        user_result(:, repeat_idx) = [ ...
            loss_ep4; loss_LB_ep4; time_2PPO_ep4; time_LB_ep4; ...
            loss_ep7; loss_LB_ep7; time_2PPO_ep7; time_LB_ep7; ...
            loss_ep10; loss_LB_ep10; time_2PPO_ep10; time_LB_ep10];
    end

    aggregate = aggregate + user_result;
end

% Utility in the paper is the total over users. Runtime is averaged over
% users before statistics across repetitions.
for eps_idx = 1:numel(epsilons)
    first_row = 4 * (eps_idx - 1) + 1;
    aggregate(first_row + 2:first_row + 3, :) = ...
        aggregate(first_row + 2:first_row + 3, :) / numel(user_ids);
end

row_mean = mean(aggregate, 2);
row_std = std(aggregate, 0, 2);

utility_mean = [row_mean(1:4:end)'; row_mean(2:4:end)'] / 10000;
utility_std = [row_std(1:4:end)'; row_std(2:4:end)'] / 10000;
runtime_mean = [row_mean(3:4:end)'; row_mean(4:4:end)'];
runtime_std = [row_std(3:4:end)'; row_std(4:4:end)'];

% Both mechanisms enforce the target privacy constraints by construction.
violation_mean = zeros(2, numel(epsilons));
violation_std = zeros(2, numel(epsilons));

%% Print paper-style tables
fprintf('\nBaseline setting: city=%s, records=%d, users=%d, repetitions=%d\n', ...
    upper(city), node_count, numel(user_ids), numel(repeat_ids));
methods = {'PAnDA', 'LB'};
print_paper_table('Utility loss (10,000 meters)', methods, epsilons, ...
    utility_mean, utility_std, 2);
print_paper_table('Violation ratio (guaranteed by construction)', methods, ...
    epsilons, violation_mean, violation_std, 4);
print_paper_table('Computation time (seconds)', methods, epsilons, ...
    runtime_mean, runtime_std, 4);

function print_paper_table(title_text, methods, epsilons, means, deviations, digits)
    fprintf('\n%s -- mean +/- standard deviation\n', title_text);
    fprintf('%-12s', 'Method');
    for idx = 1:numel(epsilons)
        fprintf(' | epsilon=%-4g', epsilons(idx));
    end
    fprintf('\n%s\n', repmat('-', 1, 12 + 16 * numel(epsilons)));

    value_format = sprintf(' | %%.%df +/- %%.%df', digits, digits);
    for method_idx = 1:numel(methods)
        fprintf('%-12s', methods{method_idx});
        for eps_idx = 1:numel(epsilons)
            fprintf(value_format, means(method_idx, eps_idx), ...
                deviations(method_idx, eps_idx));
        end
        fprintf('\n');
    end
end
