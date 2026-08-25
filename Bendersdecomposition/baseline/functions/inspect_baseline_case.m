function metadata = inspect_baseline_case( ...
        city, node_count, user_id, repeat_id, record_limit)
%INSPECT_BASELINE_CASE Read only the fields needed for resource preflight.

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
    sample = load(data_file, 'node_tar', 'obf_ID');
    if ~isfield(sample, 'node_tar') || ~isfield(sample, 'obf_ID')
        error('Baseline:MissingField', ...
            'Input file must contain node_tar and obf_ID: %s', data_file);
    end
    full_record_count = numel(sample.node_tar);
    output_count = numel(sample.obf_ID);
    if isempty(record_limit)
        effective_record_count = full_record_count;
    else
        if record_limit < output_count
            error('Baseline:RecordLimitTooSmall', ...
                'record_limit must be at least the number of outputs (%d).', ...
                output_count);
        end
        effective_record_count = min(record_limit, full_record_count);
    end
    metadata = struct('input_file', data_file, ...
        'full_record_count', full_record_count, ...
        'effective_record_count', effective_record_count, ...
        'output_count', output_count);
end
