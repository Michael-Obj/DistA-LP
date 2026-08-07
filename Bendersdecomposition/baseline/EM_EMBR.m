%% EM, RMP (Bayesian Remapping), and LP-A baseline runner
% Run this script from the baseline artifact root. Results are aggregated
% in memory and printed in paper-style tables; no result files are saved.

clear;
clc;

%% Experiment configuration
city = 'rome';              % 'rome', 'london', or 'nyc'
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

parameters;
env_parameters.NR_NODE_IN_TARGET = node_count;

nr_eps = numel(epsilons);
utility_mean = zeros(3, nr_eps);
utility_std = zeros(3, nr_eps);
runtime_mean = zeros(3, nr_eps);
runtime_std = zeros(3, nr_eps);
violation_mean = zeros(3, nr_eps);
violation_std = zeros(3, nr_eps);

%% Run every privacy budget
for eps_idx = 1:nr_eps
    epsilon = epsilons(eps_idx);
    env_parameters.EPSILON = epsilon;

    % Rows: EM loss, RMP loss, LP-A loss, EM time, RMP time,
    % LP-A time, LP-A violation ratio. Columns are repetitions.
    aggregate = zeros(7, numel(repeat_ids));

    for user_idx = 1:numel(user_ids)
        user_id = user_ids(user_idx);
        user_result = zeros(7, numel(repeat_ids));

        for repeat_idx = 1:numel(repeat_ids)
            repeat_id = repeat_ids(repeat_idx);
            data_file = fullfile(sprintf('%s_location_data_%d_nodes', city, node_count), ...
                sprintf('location_data_sample_%d', user_id), ...
                sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
            sample = load(data_file);
            node_tar = sample.node_tar;
            obf_ID = sample.obf_ID;

            distance_matrix = distanceMatrix( ...
                col_longitude(node_tar), col_latitude(node_tar));
            distance_save = distance_matrix;
            task_loc = 2;

            [loss_em, loss_rmp, ~, time_em, time_rmp] = ...
                loss_for_benchmark(env_parameters, obf_ID, distance_matrix, ...
                node_tar, G, task_loc);

            % coarse.m produces loss_coarse, time_LPCA, and vio_ratio.
            % evalc suppresses solver/intermediate output so that only the
            % final paper-style tables appear in the Command Window.
            evalc('coarse');

            user_result(:, repeat_idx) = [ ...
                loss_em; loss_rmp; loss_coarse; ...
                time_em; time_rmp; time_LPCA; vio_ratio];
        end

        aggregate = aggregate + user_result;
    end

    % Utility in the paper is the total over users. Runtime and violation
    % ratio are averaged over users before statistics across repetitions.
    aggregate(4:7, :) = aggregate(4:7, :) / numel(user_ids);

    row_mean = mean(aggregate, 2);
    row_std = std(aggregate, 0, 2);

    utility_mean(:, eps_idx) = row_mean(1:3) / 10000;
    utility_std(:, eps_idx) = row_std(1:3) / 10000;
    runtime_mean(:, eps_idx) = row_mean(4:6);
    runtime_std(:, eps_idx) = row_std(4:6);
    violation_mean(:, eps_idx) = [0; 0; row_mean(7)];
    violation_std(:, eps_idx) = [0; 0; row_std(7)];
end

%% Print paper-style tables
fprintf('\nBaseline setting: city=%s, records=%d, users=%d, repetitions=%d\n', ...
    upper(city), node_count, numel(user_ids), numel(repeat_ids));
methods = {'EM', 'RMP', 'LP-A'};
print_paper_table('Utility loss (10,000 meters)', methods, epsilons, ...
    utility_mean, utility_std, 2);
print_paper_table('Violation ratio', methods, epsilons, ...
    violation_mean, violation_std, 4);
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
