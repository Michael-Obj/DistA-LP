function case_data = load_baseline_case( ...
        city, node_count, user_id, repeat_id, record_limit)
%LOAD_BASELINE_CASE Load one common baseline case and build LP inputs.

    if nargin < 5
        record_limit = [];
    end

    baseline_root = fileparts(fileparts(mfilename('fullpath')));
    data_file = fullfile(baseline_root, ...
        sprintf('%s_location_data_%d_nodes', city, node_count), ...
        sprintf('location_data_sample_%d', user_id), ...
        sprintf('location_data_r%d_user%d.mat', repeat_id, user_id));
    if ~isfile(data_file)
        error('Baseline:MissingInput', 'Input file not found: %s', data_file);
    end

    sample = load(data_file);
    required_fields = {'node_tar', 'obf_ID'};
    for field_idx = 1:numel(required_fields)
        if ~isfield(sample, required_fields{field_idx})
            error('Baseline:MissingField', ...
                'Input file %s does not contain %s.', ...
                data_file, required_fields{field_idx});
        end
    end

    node_tar = double(sample.node_tar(:)');
    obf_ID = double(sample.obf_ID(:)');
    full_record_count = numel(node_tar);
    if any(obf_ID < 1 | obf_ID > full_record_count | obf_ID ~= floor(obf_ID))
        error('Baseline:InvalidOutputIndex', ...
            'obf_ID must contain local indices into node_tar.');
    end
    if numel(unique(obf_ID)) ~= numel(obf_ID)
        error('Baseline:DuplicateOutputIndex', ...
            'The COPT/LP interface requires unique output indices.');
    end

    % Optional small-domain mode for validating the interface. Keep every
    % supplied output and fill the remaining records deterministically.
    if ~isempty(record_limit) && record_limit < full_record_count
        if record_limit < numel(obf_ID)
            error('Baseline:RecordLimitTooSmall', ...
                'record_limit must be at least the number of outputs (%d).', ...
                numel(obf_ID));
        end
        remaining = setdiff(1:full_record_count, obf_ID, 'stable');
        selected_local = [obf_ID, ...
            remaining(1:(record_limit - numel(obf_ID)))];
        node_tar = node_tar(selected_local);
        obf_ID = 1:numel(obf_ID);
    end

    persistent cached_city cached_longitude cached_latitude cached_graph
    if isempty(cached_city) || ~strcmpi(cached_city, city)
        node_csv = fullfile(baseline_root, 'Dataset', city, 'raw', ...
            sprintf('%s_nodes.csv', city));
        opts = detectImportOptions(node_csv);
        opts = setvartype(opts, 'osmid', 'int64');
        nodes = readtable(node_csv, opts);
        cached_longitude = table2array(nodes(:, 'x'));
        cached_latitude = table2array(nodes(:, 'y'));
        graph_file = fullfile(baseline_root, sprintf('G_%s.mat', city));
        graph_data = load(graph_file, 'G');
        cached_graph = graph_data.G;
        cached_city = city;
    end

    n = numel(node_tar);
    m = numel(obf_ID);
    if n < 2
        error('Baseline:TooFewRecords', 'At least two records are required.');
    end

    distance_rr = distanceMatrix( ...
        cached_longitude(node_tar), cached_latitude(node_tar));
    distance_ro = distance_rr(:, obf_ID);

    task_loc = 2;
    [~, path_distance] = shortestpathtree(cached_graph, node_tar(task_loc));
    real_path_distance = path_distance(node_tar);
    real_path_distance = real_path_distance(:);
    output_path_distance = path_distance(node_tar(obf_ID));
    output_path_distance = output_path_distance(:)';
    cost_matrix = abs(real_path_distance - output_path_distance) / n;
    if any(~isfinite(cost_matrix), 'all')
        error('Baseline:DisconnectedGraph', ...
            'The selected case contains a non-finite road-network cost.');
    end

    case_data = struct();
    case_data.city = city;
    case_data.node_count = node_count;
    case_data.effective_node_count = n;
    case_data.user_id = user_id;
    case_data.repeat_id = repeat_id;
    case_data.node_tar = node_tar;
    case_data.obf_ID = obf_ID;
    case_data.distance_rr = distance_rr;
    case_data.distance_ro = distance_ro;
    case_data.cost_matrix = cost_matrix;
    case_data.input_file = data_file;
    case_data.output_count = m;
end
