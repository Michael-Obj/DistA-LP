addpath('./classes/Server/');
addpath('./classes/User/');
addpath('./classes/MasterProgram/');
addpath('./classes/Subproblem/');
addpath('./func/benchmarks/');
addpath('./func/benchmarks/randl/');
addpath('./func'); 
addpath('./func/read_files'); 
addpath('./func/haversine');

parameters;

% % Required fields used by User(...)
% env_parameters.LR_LOC_SIZE = 20;
% env_parameters.OBF_RANGE   = 4.0;
% env_parameters.NR_USER     = 10;

%---------- ROME 2000 -----------------
env_parameters.longitude_min = 12.2;
env_parameters.longitude_max = 12.4;
env_parameters.latitude_min  = 41.901;
env_parameters.latitude_max  = 42.10;
% --------------------------------------
% env_parameters.longitude_min = 12.601;
% env_parameters.longitude_max = 12.8;
% env_parameters.latitude_min  = 41.801;
% env_parameters.latitude_max  = 42.00;
% ----------- 4k & 6k ------------------
% env_parameters.longitude_min = 12.401;
% env_parameters.longitude_max = 12.59;
% env_parameters.latitude_min  = 41.701;
% env_parameters.latitude_max  = 41.90;
%---------- NYC 2k - 6k ----------------
% env_parameters.longitude_min = -74.3;
% env_parameters.longitude_max = -74;
% env_parameters.latitude_min  = 40.5;
% env_parameters.latitude_max  = 40.65;
%---------- LONDON 2k - 10k ------------
% env_parameters.longitude_min = -0.5;
% env_parameters.longitude_max = -0.3;
% env_parameters.latitude_min  = 51.4;
% env_parameters.latitude_max  = 51.6;

env_parameters.nr_loc_selected = 2000;
env_parameters.NEIGHBOR_THRESHOLD = 50;
env_parameters.GAMMA = 1000.0;

baseSeed = 12345;
stream   = RandStream('Threefry','Seed',baseSeed);
RandStream.setGlobalStream(stream);

env_parameters = readCityMapInfo(env_parameters);

idx_selected = randperm(numel(env_parameters.node_target), env_parameters.nr_loc_selected);
env_parameters.longitude_selected = env_parameters.longitude(idx_selected);
env_parameters.latitude_selected  = env_parameters.latitude(idx_selected);
env_parameters.node_target_selected = env_parameters.node_target(idx_selected);
env_parameters.G_mDP = mDP_graph_creator(env_parameters);

% resultsDir = fullfile('additional_experiment_results','results');
% if ~exist(resultsDir, 'dir')
%     mkdir(resultsDir);
% end

user_list = 1:env_parameters.NR_USER;
outFile = fullfile('additional_experiment_results','results','upload_communication_costs_all_surrogates.csv');
communication_cost_eval(env_parameters, user_list, outFile);
plot_communication_costs(outFile);

downloadOutFile = fullfile('additional_experiment_results','results','download_communication_costs_optimal_matrix.csv');
download_communication_cost_eval(env_parameters, user_list, downloadOutFile);
plot_download_communication_costs(downloadOutFile);

downstreamOutFile = fullfile('additional_experiment_results','results','downstream_upload_costs.csv');
n_records_per_user = 1;  % one perturbed record y_m per user
downstream_upload_cost_eval(env_parameters, user_list, n_records_per_user, downstreamOutFile);
plot_downstream_upload_costs(downstreamOutFile);


summaryOutFile = fullfile('additional_experiment_results','results', ...
    'complete_communication_cost_summary.csv');

latexOutFile = fullfile('additional_experiment_results','results', ...
    'complete_communication_cost_summary.tex');

% Choose downstream encoding:
% 'index'      = upload only y_m index
% 'coordinate' = upload longitude/latitude
% 'onehot'     = upload dense one-hot vector over Y_m
downstream_encoding = 'index';

summaryTable = build_complete_communication_table( ...
    outFile, ...
    downloadOutFile, ...
    downstreamOutFile, ...
    summaryOutFile, ...
    downstream_encoding);

write_communication_latex_table(summaryTable, latexOutFile);


function summaryTable = build_complete_communication_table( ...
    upload_csv, download_csv, downstream_csv, out_file, downstream_encoding)

    % ---------------------------------------------------------
    % This builds the final communication-cost table:
    %
    %   1. First upload: User -> Server
    %   2. Download: Server -> User
    %   3. Second upload: User -> Downstream task
    %
    % The second upload corresponds to the perturbed record y_m.
    % ---------------------------------------------------------

    uploadT     = readtable(upload_csv);
    downloadT   = readtable(download_csv);
    downstreamT = readtable(downstream_csv);

    % ---------------------------------------------------------
    % Average over users
    % ---------------------------------------------------------
    raw_upload        = mean(uploadT.raw_bytes);
    gaussian_upload   = mean(uploadT.gaussian_bytes);
    polynomial_upload = mean(uploadT.polynomial_bytes);
    rbf_upload        = mean(uploadT.rbf_bytes);
    svd_upload        = mean(uploadT.svd_bytes);

    z_download = mean(downloadT.download_bytes);

    switch lower(downstream_encoding)

        case 'index'
            downstream_upload = mean(downstreamT.index_upload_bytes);
            downstream_label = "Index y_m";

        case 'coordinate'
            downstream_upload = mean(downstreamT.coordinate_upload_bytes);
            downstream_label = "Coordinate y_m";

        case 'onehot'
            downstream_upload = mean(downstreamT.onehot_upload_bytes);
            downstream_label = "One-hot y_m";

        otherwise
            error('Unknown downstream_encoding. Use index, coordinate, or onehot.');
    end

    % ---------------------------------------------------------
    % Build table rows
    % ---------------------------------------------------------
    Representation = [
        "Raw matrices";
        "Gaussian";
        "Polynomial";
        "RBF";
        "SVD"
    ];

    upload_to_server_bytes = [
        raw_upload;
        gaussian_upload;
        polynomial_upload;
        rbf_upload;
        svd_upload
    ];

    download_from_server_bytes = repmat(z_download, 5, 1);

    second_upload_to_downstream_bytes = repmat(downstream_upload, 5, 1);

    total_end_to_end_bytes = ...
        upload_to_server_bytes + ...
        download_from_server_bytes + ...
        second_upload_to_downstream_bytes;

    % ---------------------------------------------------------
    % Compression ratios
    % ---------------------------------------------------------
    upload_compression = ...
        upload_to_server_bytes(1) ./ upload_to_server_bytes;

    end_to_end_compression = ...
        total_end_to_end_bytes(1) ./ total_end_to_end_bytes;

    % ---------------------------------------------------------
    % Convert to KB also
    % ---------------------------------------------------------
    upload_to_server_kb = upload_to_server_bytes / 1024;
    download_from_server_kb = download_from_server_bytes / 1024;
    second_upload_to_downstream_kb = second_upload_to_downstream_bytes / 1024;
    total_end_to_end_kb = total_end_to_end_bytes / 1024;

    downstream_encoding_used = repmat(downstream_label, 5, 1);

    % ---------------------------------------------------------
    % Final table
    % ---------------------------------------------------------
    summaryTable = table( ...
        Representation, ...
        upload_to_server_bytes, ...
        download_from_server_bytes, ...
        second_upload_to_downstream_bytes, ...
        total_end_to_end_bytes, ...
        upload_compression, ...
        end_to_end_compression, ...
        upload_to_server_kb, ...
        download_from_server_kb, ...
        second_upload_to_downstream_kb, ...
        total_end_to_end_kb, ...
        downstream_encoding_used, ...
        'VariableNames', { ...
        'Representation', ...
        'upload_to_server_bytes', ...
        'download_from_server_bytes', ...
        'second_upload_to_downstream_bytes', ...
        'total_end_to_end_bytes', ...
        'upload_compression', ...
        'end_to_end_compression', ...
        'upload_to_server_kb', ...
        'download_from_server_kb', ...
        'second_upload_to_downstream_kb', ...
        'total_end_to_end_kb', ...
        'downstream_encoding'});

    disp(summaryTable);

    if ~isempty(out_file)
        resultsDir = fileparts(out_file);
        if ~exist(resultsDir, 'dir')
            mkdir(resultsDir);
        end
        writetable(summaryTable, out_file);
    end

end


function write_communication_latex_table(summaryTable, latex_file)

    % ---------------------------------------------------------
    % Writes a compact LaTeX table for the paper.
    % Columns:
    %   Representation
    %   Upload to server
    %   Download from server
    %   Upload to downstream
    %   Total
    %   Upload compression
    %   End-to-end compression
    % ---------------------------------------------------------

    resultsDir = fileparts(latex_file);
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end

    fid = fopen(latex_file, 'w');

    fprintf(fid, '\\begin{table}[t]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{End-to-end communication cost of DISTA-LP.}\n');
    fprintf(fid, '\\label{tab:communication-cost}\n');
    fprintf(fid, '\\begin{tabular}{lrrrrrr}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, 'Representation & Upload & Download & Second Upload & Total & Upload Comp. & End-to-End Comp. \\\\\n');
    fprintf(fid, '\\midrule\n');

    for i = 1:height(summaryTable)

        fprintf(fid, '%s & %.0f & %.0f & %.0f & %.0f & %.2f$\\times$ & %.2f$\\times$ \\\\\n', ...
            summaryTable.Representation(i), ...
            summaryTable.upload_to_server_bytes(i), ...
            summaryTable.download_from_server_bytes(i), ...
            summaryTable.second_upload_to_downstream_bytes(i), ...
            summaryTable.total_end_to_end_bytes(i), ...
            summaryTable.upload_compression(i), ...
            summaryTable.end_to_end_compression(i));
    end

    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');

    fclose(fid);

end






function communication_cost_eval(env_parameters, user_indices, out_file)

    sfloat = 8;  % bytes per double

    n_users = length(user_indices);

    raw_bytes        = zeros(n_users,1);
    gaussian_bytes   = zeros(n_users,1);
    polynomial_bytes = zeros(n_users,1);
    rbf_bytes        = zeros(n_users,1);
    svd_bytes        = zeros(n_users,1);

    gaussian_ratio   = zeros(n_users,1);
    polynomial_ratio = zeros(n_users,1);
    rbf_ratio        = zeros(n_users,1);
    svd_ratio        = zeros(n_users,1);

    for idx = 1:n_users

        uid = user_indices(idx);

        u = User(uid, env_parameters.LR_LOC_SIZE, env_parameters.OBF_RANGE, env_parameters.NEIGHBOR_THRESHOLD, env_parameters);            
        u = u.initialization(env_parameters);
        % u = u.lowrank_SVD_fit(env_parameters);
        % u = u.lowrank_SVD_noisy_parameters(env_parameters);
        % u = u.lowrank_SVD_recover(env_parameters);

        % ---------------------------------------------------------
        % Raw matrices: A_dx, A_dy, A_cy
        % ---------------------------------------------------------
        n_LR   = numel(u.distance_matrix_LR);
        n_LR2o = numel(u.distance_matrix_LR2obf);
        n_cost = numel(u.cost_matrix_RL);

        raw_bytes(idx) = sfloat * (n_LR + n_LR2o + n_cost);

        % ---------------------------------------------------------
        % Gaussian surrogate communication cost
        % initialization() already runs gaussian_fit,
        % gaussian_noisy_parameters, and gaussian_recover in your User.m.
        % ---------------------------------------------------------
        n_gaussian_params = ...
            count_numeric_values(u.fitted_best_params_1) + ...
            count_numeric_values(u.fitted_best_params_2) + ...
            count_numeric_values(u.fitted_best_params_3) + ...
            count_numeric_values(u.best_pi);

        gaussian_bytes(idx) = sfloat * n_gaussian_params;
        gaussian_ratio(idx) = raw_bytes(idx) / max(1, gaussian_bytes(idx));

        % ---------------------------------------------------------
        % Polynomial surrogate communication cost
        % ---------------------------------------------------------
        u_poly = u;
        try
            if ismethod(u_poly, 'polynomial_fit')
                u_poly = u_poly.polynomial_fit(env_parameters);
            end
            if ismethod(u_poly, 'polynomial_noisy_parameters')
                u_poly = u_poly.polynomial_noisy_parameters(env_parameters);
            end
            if ismethod(u_poly, 'polynomial_recover')
                u_poly = polynomial_recover(u_poly, env_parameters);
            end
        catch ME
            warning('Polynomial surrogate failed for user %d: %s', uid, ME.message);
        end

        n_poly_params = ...
            count_numeric_values(u_poly.fitted_best_coeffs_1) + ...
            count_numeric_values(u_poly.fitted_best_coeffs_2) + ...
            count_numeric_values(u_poly.fitted_best_coeffs_3) + ...
            count_numeric_values(u_poly.best_pi_poly);

        polynomial_bytes(idx) = sfloat * n_poly_params;
        polynomial_ratio(idx) = raw_bytes(idx) / max(1, polynomial_bytes(idx));

        % ---------------------------------------------------------
        % RBF surrogate communication cost
        % ---------------------------------------------------------
        u_rbf = u;
        try
            if ismethod(u_rbf, 'RBF_fit')
                u_rbf = u_rbf.RBF_fit(env_parameters);
            end
            if ismethod(u_rbf, 'RBF_noisy_parameters')
                u_rbf = u_rbf.RBF_noisy_parameters(env_parameters);
            end
            if ismethod(u_rbf, 'RBF_recover')
                u_rbf = RBF_recover(u_rbf, env_parameters);
            end
        catch ME
            warning('RBF surrogate failed for user %d: %s', uid, ME.message);
        end

        n_rbf_params = ...
            count_numeric_values(u_rbf.fitted_best_w_1) + ...
            count_numeric_values(u_rbf.fitted_best_w_2) + ...
            count_numeric_values(u_rbf.fitted_best_w_3) + ...
            count_numeric_values(u_rbf.best_pi_rbf);

        % If your RBF centers are transmitted instead of fixed globally,
        % uncomment this line:
        % n_rbf_params = n_rbf_params + 2 * env_parameters.NUM_CENTRES;

        rbf_bytes(idx) = sfloat * n_rbf_params;
        rbf_ratio(idx) = raw_bytes(idx) / max(1, rbf_bytes(idx));

        % % ---------------------------------------------------------
        % % Low-rank SVD surrogate communication cost
        % % ---------------------------------------------------------
        % r = env_parameters.rank_r;
        % 
        % A1 = u.distance_matrix_LR;
        % A2 = u.distance_matrix_LR2obf;
        % A3 = u.cost_matrix_RL;
        % 
        % n_svd_params = svd_param_count(A1, r) + ...
        %                svd_param_count(A2, r) + ...
        %                svd_param_count(A3, r);
        % 
        % % Add the permutation vector if it must be uploaded.
        % n_svd_params = n_svd_params + env_parameters.LR_LOC_SIZE;
        % 
        % svd_bytes(idx) = sfloat * n_svd_params;
        % svd_ratio(idx) = raw_bytes(idx) / max(1, svd_bytes(idx));

        [U,S,V] = svds(u.distance_matrix_LR, env_parameters.rank_r);  % top‑r SVD
        u.fitted_best_factors_struct.U = U;
        u.fitted_best_factors_struct.S = diag(S);
        u.fitted_best_factors_struct.V = V;

        if isprop(u, 'fitted_best_factors_struct') && ~isempty(u.fitted_best_factors_struct)
            factors = u.fitted_best_factors_struct;
            % U is (LR×r), S is (r×r) diagonal, V is (|obf|×r)
            n_svd_params = numel(factors.U) + numel(factors.S) + numel(factors.V);
        else
            % Fallback: approximate using vectors of factor lengths
            n_svd_params = numel(u.fitted_best_factors_1) + numel(u.fitted_best_factors_2) + numel(u.fitted_best_factors_3);
        end
        svd_bytes(idx) = sfloat * n_svd_params;
        svd_ratio(idx) = raw_bytes(idx) / max(1, svd_bytes(idx));

    end

    results = table(user_indices(:), ...
        raw_bytes, ...
        gaussian_bytes, polynomial_bytes, rbf_bytes, svd_bytes, ...
        gaussian_ratio, polynomial_ratio, rbf_ratio, svd_ratio, ...
        'VariableNames', { ...
        'user_id', ...
        'raw_bytes', ...
        'gaussian_bytes', 'polynomial_bytes', 'rbf_bytes', 'svd_bytes', ...
        'gaussian_compression_ratio', 'polynomial_compression_ratio', ...
        'rbf_compression_ratio', 'svd_compression_ratio'});

    disp(results);

    if ~isempty(out_file)
        resultsDir = fileparts(out_file);
        if ~exist(resultsDir, 'dir')
            mkdir(resultsDir);
        end
        writetable(results, out_file);
    end
end


function n = svd_param_count(A, r)
    [m, k] = size(A);
    r_eff = min([r, m, k]);

    % U: m x r
    % S: r singular values
    % V: k x r
    n = r_eff * (m + k + 1);
end


function n = count_numeric_values(x)

    n = 0;

    if isempty(x)
        return;
    end

    if isnumeric(x) || islogical(x)
        n = numel(x);
        return;
    end

    if iscell(x)
        for i = 1:numel(x)
            n = n + count_numeric_values(x{i});
        end
        return;
    end

    if isstruct(x)
        f = fieldnames(x);
        for i = 1:numel(f)
            n = n + count_numeric_values(x.(f{i}));
        end
        return;
    end
end


function plot_communication_costs(csv_file)

    results = readtable(csv_file);

    user_id = results.user_id;

    raw_kb        = results.raw_bytes / 1024;
    gaussian_kb   = results.gaussian_bytes / 1024;
    polynomial_kb = results.polynomial_bytes / 1024;
    rbf_kb        = results.rbf_bytes / 1024;
    svd_kb        = results.svd_bytes / 1024;

    plot_dir = fullfile('additional_experiment_results','results','plots');
    if ~exist(plot_dir, 'dir')
        mkdir(plot_dir);
    end

    figure('Color','w','Position',[100 100 1200 750]);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    % ---------------------------------------------------------
    % Plot 1: user-wise communication cost
    % ---------------------------------------------------------
    nexttile;
    plot(user_id, raw_kb, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on;
    plot(user_id, gaussian_kb, '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(user_id, polynomial_kb, '-^', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(user_id, rbf_kb, '-d', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(user_id, svd_kb, '-x', 'LineWidth', 1.5, 'MarkerSize', 4);
    hold off;

    xlabel('User ID');
    ylabel('Communication Cost (KB)');
    title('User-wise Communication Cost');
    legend('Raw', 'Gaussian', 'Polynomial', 'RBF', 'SVD', 'Location', 'best');
    grid on;

    % ---------------------------------------------------------
    % Plot 2: average communication cost
    % ---------------------------------------------------------
    nexttile;
    avg_costs = [ ...
        mean(raw_kb), ...
        mean(gaussian_kb), ...
        mean(polynomial_kb), ...
        mean(rbf_kb), ...
        mean(svd_kb)];

    bar(avg_costs);
    set(gca, 'XTickLabel', {'Raw','Gaussian','Polynomial','RBF','SVD'});
    ylabel('Average Communication Cost (KB)');
    title('Average Communication Cost');
    grid on;

    % ---------------------------------------------------------
    % Plot 3: average compression ratio
    % ---------------------------------------------------------
    nexttile;
    avg_ratios = [ ...
        mean(results.gaussian_compression_ratio), ...
        mean(results.polynomial_compression_ratio), ...
        mean(results.rbf_compression_ratio), ...
        mean(results.svd_compression_ratio)];

    bar(avg_ratios);
    set(gca, 'XTickLabel', {'Gaussian','Polynomial','RBF','SVD'});
    ylabel('Compression Ratio');
    title('Average Compression Ratio');
    grid on;

    % ---------------------------------------------------------
    % Plot 4: surrogate-only comparison
    % ---------------------------------------------------------
    nexttile;
    surrogate_costs = [ ...
        mean(gaussian_kb), ...
        mean(polynomial_kb), ...
        mean(rbf_kb), ...
        mean(svd_kb)];

    bar(surrogate_costs);
    set(gca, 'XTickLabel', {'Gaussian','Polynomial','RBF','SVD'});
    ylabel('Average Surrogate Cost (KB)');
    title('Surrogate Upload Size Only');
    grid on;

    sgtitle('Communication-Cost Evaluation Across Surrogate Families');

    saveas(gcf, fullfile(plot_dir, 'communication_cost_all_surrogates.png'));
    savefig(gcf, fullfile(plot_dir, 'communication_cost_all_surrogates.fig'));

end












function download_communication_cost_eval(env_parameters, user_indices, out_file)

    sfloat = 8;  % bytes per double

    n_users = length(user_indices);

    LR_size        = zeros(n_users,1);
    obf_size       = zeros(n_users,1);
    z_entries      = zeros(n_users,1);
    download_bytes = zeros(n_users,1);
    download_kb    = zeros(n_users,1);

    for idx = 1:n_users

        uid = user_indices(idx);

        u = User(uid, env_parameters.LR_LOC_SIZE, ...
                 env_parameters.OBF_RANGE, ...
                 env_parameters.NEIGHBOR_THRESHOLD, ...
                 env_parameters);

        u = u.initialization(env_parameters);

        % ---------------------------------------------------------
        % Optimal perturbation matrix Z_m has size:
        % |local relevant locations| x |obfuscation locations|
        %
        % This matches the server-to-user download in the framework.
        % ---------------------------------------------------------
        n_LR  = size(u.cost_matrix_RL, 1);
        n_obf = size(u.cost_matrix_RL, 2);

        LR_size(idx)   = n_LR;
        obf_size(idx)  = n_obf;
        z_entries(idx) = n_LR * n_obf;

        % Dense perturbation matrix download cost
        download_bytes(idx) = sfloat * z_entries(idx);
        download_kb(idx)    = download_bytes(idx) / 1024;

    end

    results = table(user_indices(:), LR_size, obf_size, z_entries, ...
                    download_bytes, download_kb, ...
        'VariableNames', {'user_id','LR_size','obf_size','Z_entries', ...
                          'download_bytes','download_kb'});

    disp(results);

    if ~isempty(out_file)
        resultsDir = fileparts(out_file);
        if ~exist(resultsDir, 'dir')
            mkdir(resultsDir);
        end
        writetable(results, out_file);
    end

end


function plot_download_communication_costs(csv_file)

    results = readtable(csv_file);

    user_id = results.user_id;
    download_kb = results.download_kb;

    plot_dir = fullfile('additional_experiment_results','results','plots');
    if ~exist(plot_dir, 'dir')
        mkdir(plot_dir);
    end

    figure('Color','w','Position',[100 100 1000 600]);

    plot(user_id, download_kb, '-o', ...
         'LineWidth', 1.7, ...
         'MarkerSize', 5);

    xlabel('User ID');
    ylabel('Download Communication Cost (KB)');
    title('Server-to-User Download Cost of Optimal Perturbation Matrix');
    grid on;

    saveas(gcf, fullfile(plot_dir, 'download_cost_optimal_perturbation_matrix.png'));
    savefig(gcf, fullfile(plot_dir, 'download_cost_optimal_perturbation_matrix.fig'));

end











function downstream_upload_cost_eval(env_parameters, user_indices, n_records_per_user, out_file)

    % Communication cost from Record Perturbation block to Downstream Task.
    %
    % This measures the cost of uploading the perturbed record y_m.
    % It is different from:
    %   1. coefficient/surrogate upload to the server
    %   2. perturbation matrix Z_m download from the server
    %
    % Here we report three possible encodings of y_m:
    %   index encoding      : send obfuscated location index only
    %   coordinate encoding : send longitude/latitude pair
    %   one-hot encoding    : send dense |Y_m|-dimensional vector

    sfloat = 8;  % bytes per double
    sint   = 8;  % bytes per integer/index, assuming int64 or double index

    n_users = length(user_indices);

    LR_size = zeros(n_users,1);
    obf_size = zeros(n_users,1);

    index_upload_bytes      = zeros(n_users,1);
    coordinate_upload_bytes = zeros(n_users,1);
    onehot_upload_bytes     = zeros(n_users,1);

    index_upload_kb      = zeros(n_users,1);
    coordinate_upload_kb = zeros(n_users,1);
    onehot_upload_kb     = zeros(n_users,1);

    for idx = 1:n_users

        uid = user_indices(idx);

        u = User(uid, env_parameters.LR_LOC_SIZE, ...
                 env_parameters.OBF_RANGE, ...
                 env_parameters.NEIGHBOR_THRESHOLD, ...
                 env_parameters);

        u = u.initialization(env_parameters);

        % The downstream perturbed record y_m belongs to the obfuscation set Y_m.
        % Therefore |Y_m| is the number of obfuscation candidates.
        n_LR  = size(u.cost_matrix_RL, 1);
        n_obf = size(u.cost_matrix_RL, 2);

        LR_size(idx)  = n_LR;
        obf_size(idx) = n_obf;

        % ---------------------------------------------------------
        % Encoding 1: send only the selected obfuscated location index.
        % Cost per record = one integer.
        % ---------------------------------------------------------
        index_upload_bytes(idx) = n_records_per_user * sint;

        % ---------------------------------------------------------
        % Encoding 2: send selected obfuscated coordinate pair.
        % Cost per record = longitude + latitude = 2 doubles.
        % ---------------------------------------------------------
        coordinate_upload_bytes(idx) = n_records_per_user * 2 * sfloat;

        % ---------------------------------------------------------
        % Encoding 3: send one-hot vector over all obfuscation locations.
        % Cost per record = |Y_m| doubles.
        % ---------------------------------------------------------
        onehot_upload_bytes(idx) = n_records_per_user * n_obf * sfloat;

        index_upload_kb(idx)      = index_upload_bytes(idx) / 1024;
        coordinate_upload_kb(idx) = coordinate_upload_bytes(idx) / 1024;
        onehot_upload_kb(idx)     = onehot_upload_bytes(idx) / 1024;

    end

    results = table(user_indices(:), LR_size, obf_size, ...
        repmat(n_records_per_user,n_users,1), ...
        index_upload_bytes, coordinate_upload_bytes, onehot_upload_bytes, ...
        index_upload_kb, coordinate_upload_kb, onehot_upload_kb, ...
        'VariableNames', { ...
        'user_id', 'LR_size', 'obf_size', 'n_records', ...
        'index_upload_bytes', 'coordinate_upload_bytes', 'onehot_upload_bytes', ...
        'index_upload_kb', 'coordinate_upload_kb', 'onehot_upload_kb'});

    disp(results);

    if ~isempty(out_file)
        resultsDir = fileparts(out_file);
        if ~exist(resultsDir, 'dir')
            mkdir(resultsDir);
        end
        writetable(results, out_file);
    end

end


function plot_downstream_upload_costs(csv_file)

    results = readtable(csv_file);

    user_id = results.user_id;

    index_kb      = results.index_upload_kb;
    coordinate_kb = results.coordinate_upload_kb;
    onehot_kb     = results.onehot_upload_kb;

    plot_dir = fullfile('additional_experiment_results','results','plots');
    if ~exist(plot_dir, 'dir')
        mkdir(plot_dir);
    end

    figure('Color','w','Position',[100 100 1100 650]);

    plot(user_id, index_kb, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    hold on;
    plot(user_id, coordinate_kb, '-s', 'LineWidth', 1.5, 'MarkerSize', 5);
    plot(user_id, onehot_kb, '-^', 'LineWidth', 1.5, 'MarkerSize', 5);
    hold off;

    xlabel('User ID');
    ylabel('Downstream Upload Cost (KB)');
    title('User-to-Downstream Upload Cost of Perturbed Record y_m');
    legend('Index encoding', 'Coordinate encoding', 'One-hot encoding', ...
           'Location', 'best');
    grid on;

    saveas(gcf, fullfile(plot_dir, 'downstream_upload_cost_perturbed_record.png'));
    savefig(gcf, fullfile(plot_dir, 'downstream_upload_cost_perturbed_record.fig'));

end
