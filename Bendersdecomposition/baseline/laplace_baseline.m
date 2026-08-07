%% Planar Laplace baseline runner
% Run this script from the baseline artifact root. Results are aggregated
% in memory and printed in paper-style tables; no result files are saved.

clear;
clc;

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

%% Load the selected city once
node_csv = sprintf('./Dataset/%s/raw/%s_nodes.csv', city, city);
edge_csv = sprintf('./Dataset/%s/raw/%s_edges.csv', city, city);
opts = detectImportOptions(node_csv);
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_csv, opts);
df_edges = readtable(edge_csv); %#ok<NASGU>
col_longitude = table2array(df_nodes(:, 'x'));
col_latitude = table2array(df_nodes(:, 'y'));

graph_data = load(sprintf('G_%s.mat', city), 'G');
G = graph_data.G;

% Rows for each epsilon: utility loss, runtime, violation ratio.
aggregate = zeros(3 * numel(epsilons), numel(repeat_ids));

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

        if isfield(sample, 'lon_sel')
            loc_lons = sample.lon_sel;
        else
            loc_lons = col_longitude(node_tar);
        end
        if isfield(sample, 'lat_sel')
            loc_lats = sample.lat_sel;
        else
            loc_lats = col_latitude(node_tar);
        end

        pert_lons = col_longitude(obf_ID');
        pert_lats = col_latitude(obf_ID');

        nr_locations = numel(node_tar);
        nr_outputs = numel(obf_ID);
        cost_matrix = zeros(nr_locations, nr_outputs);
        [~, path_distance] = shortestpathtree(G, node_tar(2));
        for location_idx = 1:nr_locations
            for output_idx = 1:nr_outputs
                cost_matrix(location_idx, output_idx) = abs( ...
                    path_distance(node_tar(location_idx)) - ...
                    path_distance(node_tar(obf_ID(output_idx))));
            end
        end
        cost_matrix = cost_matrix / nr_locations;

        for eps_idx = 1:numel(epsilons)
            epsilon = epsilons(eps_idx);
            [mechanism, ~, runtime] = planar_laplace_utility_loss( ...
                loc_lons, loc_lats, pert_lons, pert_lats, ...
                cost_matrix, epsilon);
            utility_loss = sum(sum(cost_matrix .* mechanism));

            % Planar Laplace satisfies the target privacy definition by
            % construction, so its violation ratio is zero.
            violation_ratio = 0;
            first_row = 3 * (eps_idx - 1) + 1;
            user_result(first_row:first_row + 2, repeat_idx) = ...
                [utility_loss; runtime; violation_ratio];
        end
    end

    aggregate = aggregate + user_result;
end

% Utility in the paper is the total over users. Runtime and violation ratio
% are averaged over users before statistics across repetitions.
for eps_idx = 1:numel(epsilons)
    first_row = 3 * (eps_idx - 1) + 1;
    aggregate(first_row + 1:first_row + 2, :) = ...
        aggregate(first_row + 1:first_row + 2, :) / numel(user_ids);
end

row_mean = mean(aggregate, 2);
row_std = std(aggregate, 0, 2);

utility_mean = row_mean(1:3:end)' / 10000;
utility_std = row_std(1:3:end)' / 10000;
runtime_mean = row_mean(2:3:end)';
runtime_std = row_std(2:3:end)';
violation_mean = row_mean(3:3:end)';
violation_std = row_std(3:3:end)';

%% Print paper-style tables
fprintf('\nBaseline setting: city=%s, records=%d, users=%d, repetitions=%d\n', ...
    upper(city), node_count, numel(user_ids), numel(repeat_ids));
methods = {'Laplace'};
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
