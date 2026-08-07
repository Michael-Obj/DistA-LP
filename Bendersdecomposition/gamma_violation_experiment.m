addpath('./classes/Server/');
addpath('./classes/User/');
addpath('./classes/MasterProgram/');
addpath('./classes/Subproblem/');
addpath('./func/benchmarks/');
addpath('./func/benchmarks/randl/');
addpath('./func'); 
addpath('./func/read_files'); 
addpath('./func/haversine');

clear;
clc;

parameters;

% ---------------------------------------------------------
% Required settings
% ---------------------------------------------------------
env_parameters.LR_LOC_SIZE = 20;
env_parameters.OBF_RANGE   = 4.0;
env_parameters.NR_USER     = 10;
env_parameters.rank_r      = 2;

%---------- ROME 2000 -----------------
% env_parameters.longitude_min = 12.2;
% env_parameters.longitude_max = 12.4;
% env_parameters.latitude_min  = 41.901;
% env_parameters.latitude_max  = 42.10;
% --------------------------------------
% env_parameters.longitude_min = 12.601;
% env_parameters.longitude_max = 12.8;
% env_parameters.latitude_min  = 41.801;
% env_parameters.latitude_max  = 42.00;
% ----------- 4k & 6k ------------------
env_parameters.longitude_min = 12.401;
env_parameters.longitude_max = 12.59;
env_parameters.latitude_min  = 41.701;
env_parameters.latitude_max  = 41.90;
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

env_parameters.nr_loc_selected = 4000;
env_parameters.NEIGHBOR_THRESHOLD = 50;
env_parameters.GAMMA = 1000.0;
env_parameters.EPSILON = 10;
paper_dista_lp_violation = 0.00;

baseSeed = 12345;
stream = RandStream('Threefry','Seed',baseSeed);
RandStream.setGlobalStream(stream);

env_parameters = readCityMapInfo(env_parameters);

resultsDir = fullfile('additional_experiment_results','results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Repair parameters to test
gamma_values = [0, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5];

user_list = 1:env_parameters.NR_USER;

outFile = fullfile(resultsDir, 'gamma_vs_violation_svd.csv');

gamma_vs_violation_svd(env_parameters, user_list, gamma_values, outFile);

plot_gamma_violation(outFile);


function gamma_vs_violation_svd(env_parameters, user_indices, gamma_values, out_file)

    n_users = length(user_indices);
    n_gamma = length(gamma_values);

    rows = [];

    for uidx = 1:n_users

        uid = user_indices(uidx);

        % Different deterministic random substream per user
        stream = RandStream('Threefry','Seed',12345);
        stream.Substream = uid;
        RandStream.setGlobalStream(stream);
        
        available_nodes = numel(env_parameters.node_target);
        n_selected = min(env_parameters.nr_loc_selected, available_nodes);
        
        % Important: make idx_selected a column vector
        idx_selected = randperm(available_nodes, n_selected).';
        
        env_user = env_parameters;
        
        % Important: force longitude and latitude to be column vectors
        env_user.longitude_selected = env_user.longitude(idx_selected);
        env_user.latitude_selected  = env_user.latitude(idx_selected);
        env_user.node_target_selected = env_user.node_target(idx_selected);
        
        env_user.longitude_selected = env_user.longitude_selected(:);
        env_user.latitude_selected  = env_user.latitude_selected(:);
        env_user.node_target_selected = env_user.node_target_selected(:);
        
        % Important: keep nr_loc_selected consistent after min(...)
        env_user.nr_loc_selected = numel(env_user.longitude_selected);
        
        % Quick sanity check before graph creation
        fprintf('User %d: longitude_selected = %d x %d, latitude_selected = %d x %d, nr_loc_selected = %d\n', ...
            uid, ...
            size(env_user.longitude_selected,1), size(env_user.longitude_selected,2), ...
            size(env_user.latitude_selected,1), size(env_user.latitude_selected,2), ...
            env_user.nr_loc_selected);
        
        env_user.G_mDP = mDP_graph_creator(env_user);


        if env_user.nr_loc_selected < uid
            error('Cannot create user %d because only %d selected nodes are available.', ...
                uid, env_user.nr_loc_selected);
        end
        
        if isempty(env_user.longitude_selected) || isempty(env_user.latitude_selected)
            error('Selected longitude/latitude arrays are empty for user %d.', uid);
        end

        u = User(uid, ...
                 env_user.LR_LOC_SIZE, ...
                 env_user.OBF_RANGE, ...
                 env_user.NEIGHBOR_THRESHOLD, ...
                 env_user);

        u = u.initialization(env_user);

        % ---------------------------------------------------------
        % Use SVD reconstruction
        % ---------------------------------------------------------
        u = u.lowrank_SVD_fit(env_user);
        u = u.lowrank_SVD_noisy_parameters(env_user);
        u = u.lowrank_SVD_recover(env_user);

        % True and reconstructed matrices
        D1_true = u.distance_matrix_LR;
        D1_hat  = u.distance_matrix_LR_recovered_s;

        D2_true = u.distance_matrix_LR2obf;
        D2_hat  = u.distance_matrix_LR2obf_recovered_s;

        % Maximum one-sided overestimation
        max_over_D1 = max(max(D1_hat - D1_true, 0), [], 'all');
        max_over_D2 = max(max(D2_hat - D2_true, 0), [], 'all');

        for gidx = 1:n_gamma

            gamma = gamma_values(gidx);

            % Repair reconstructed distances
            D1_repaired = max(D1_hat - gamma, 0);
            D2_repaired = max(D2_hat - gamma, 0);

            % Violation ratio after repair
            viol_D1 = violation_ratio(D1_true, D1_repaired);
            viol_D2 = violation_ratio(D2_true, D2_repaired);

            % Relative error after repair
            rel_D1 = relative_error(D1_true, D1_repaired);
            rel_D2 = relative_error(D2_true, D2_repaired);

            % Utility loss proxy
            abs_D1 = mean(abs(D1_true(:) - D1_repaired(:)));
            abs_D2 = mean(abs(D2_true(:) - D2_repaired(:)));

            rows = [rows; ...
                uid, gamma, ...
                max_over_D1, max_over_D2, ...
                viol_D1, viol_D2, ...
                rel_D1, rel_D2, ...
                abs_D1, abs_D2];

        end

    end

    results = array2table(rows, ...
        'VariableNames', { ...
        'user_id', 'gamma', ...
        'max_over_DLR', 'max_over_DLR2obf', ...
        'viol_DLR', 'viol_DLR2obf', ...
        'relerr_DLR', 'relerr_DLR2obf', ...
        'abs_error_DLR', 'abs_error_DLR2obf'});

    disp(results);

    writetable(results, out_file);

    % ---------------------------------------------------------
    % Summary over users
    % ---------------------------------------------------------
    summaryRows = [];

    uniqueGammas = unique(results.gamma);

    for i = 1:length(uniqueGammas)

        gamma = uniqueGammas(i);

        T = results(results.gamma == gamma, :);

        summaryRows = [summaryRows; ...
            gamma, ...
            mean(T.viol_DLR), std(T.viol_DLR), ...
            mean(T.viol_DLR2obf), std(T.viol_DLR2obf), ...
            mean(T.relerr_DLR), std(T.relerr_DLR), ...
            mean(T.relerr_DLR2obf), std(T.relerr_DLR2obf), ...
            mean(T.abs_error_DLR), std(T.abs_error_DLR), ...
            mean(T.abs_error_DLR2obf), std(T.abs_error_DLR2obf)];

    end

    summary = array2table(summaryRows, ...
        'VariableNames', { ...
        'gamma', ...
        'mean_viol_DLR', 'std_viol_DLR', ...
        'mean_viol_DLR2obf', 'std_viol_DLR2obf', ...
        'mean_relerr_DLR', 'std_relerr_DLR', ...
        'mean_relerr_DLR2obf', 'std_relerr_DLR2obf', ...
        'mean_abs_error_DLR', 'std_abs_error_DLR', ...
        'mean_abs_error_DLR2obf', 'std_abs_error_DLR2obf'});

    summaryFile = strrep(out_file, '.csv', '_summary.csv');
    writetable(summary, summaryFile);

    disp('Summary over users:');
    disp(summary);

end

function plot_gamma_violation(csv_file)

    % Read detailed and summary files
    results = readtable(csv_file);

    summaryFile = strrep(csv_file, '.csv', '_summary.csv');
    summary = readtable(summaryFile);

    plotDir = fullfile('additional_experiment_results','results','plots');
    if ~exist(plotDir, 'dir')
        mkdir(plotDir);
    end

    gamma = summary.gamma;

    % =========================================================
    % Plot 1: Gamma vs violation ratio for D_LR only
    % =========================================================
    figure('Color','w','Position',[100 100 900 600]);

    plot(gamma, summary.mean_viol_DLR, '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;

    % % Original unrepaired baseline at gamma = 0
    % original_DLR_viol = summary.mean_viol_DLR(summary.gamma == 0);
    % if ~isempty(original_DLR_viol)
    %     yline(original_DLR_viol, '--', ...
    %         sprintf('Original \\gamma=0: %.4f', original_DLR_viol), ...
    %         'LineWidth', 1.2);
    % end
    % ------------------------------------------------------------
    % % Original unrepaired reconstruction violation at gamma = 0
    original_DLR_viol = summary.mean_viol_DLR(summary.gamma == 0);
    
    % if ~isempty(original_DLR_viol)
    %     yline(original_DLR_viol, '--', ...
    %         sprintf('Unrepaired reconstruction \\gamma=0: %.4f', original_DLR_viol), ...
    %         'LineWidth', 1.2);
    % end
    
    % % DistA-LP downstream violation ratio reported in Table III
    paper_dista_lp_violation = 0.00;
    
    yline(paper_dista_lp_violation, ':', ...
        'LineWidth', 1.5);

    hold off;
    xlabel('\gamma repair parameter');
    ylabel('Violation ratio');
    title('\gamma vs mDP Distance-Overestimation Violation Ratio for D_{LR}');
    legend('Reconstructed D_{LR}', 'Original Violation Ratio', 'Location', 'best');
    grid on;

    saveas(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio_DLR.png'));
    savefig(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio_DLR.fig'));


    % =========================================================
    % Plot 2: Gamma vs violation ratio for D_LR2obf only
    % =========================================================
    figure('Color','w','Position',[100 100 900 600]);

    plot(gamma, summary.mean_viol_DLR2obf, '-s', ...
        'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;

    % % Original unrepaired reconstruction violation at gamma = 0
    original_DLR_viol = summary.mean_viol_DLR(summary.gamma == 0);
    
    % if ~isempty(original_DLR_viol)
    %     yline(original_DLR_viol, '--', ...
    %         sprintf('Unrepaired reconstruction \\gamma=0: %.4f', original_DLR_viol), ...
    %         'LineWidth', 1.2);
    % end
    
    % % DistA-LP downstream violation ratio reported in Table III
    paper_dista_lp_violation = 0.00;
    
    yline(paper_dista_lp_violation, ':', ...
        'LineWidth', 1.5);

    hold off;
    xlabel('\gamma repair parameter');
    ylabel('Violation ratio');
    title('\gamma vs mDP Distance-Overestimation Violation Ratio for D_{LR2obf}');
    legend('Reconstructed D_{LR2obf}', 'Original Violation Ratio', 'Location', 'best');
    grid on;

    saveas(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio_DLR2obf.png'));
    savefig(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio_DLR2obf.fig'));


    % =========================================================
    % Plot 3: Gamma vs reconstruction error for D_LR only
    % =========================================================
    figure('Color','w','Position',[100 100 900 600]);
    
    plot(gamma, summary.mean_relerr_DLR, '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;

    % Original unrepaired baseline at gamma = 0
    original_DLR_viol = summary.mean_viol_DLR(summary.gamma == 0);
    if ~isempty(original_DLR_viol)
        yline(original_DLR_viol, '--', ...
            sprintf('Original \\gamma=0: %.4f', original_DLR_viol), ...
            'LineWidth', 1.2);
    end

    hold off;
    xlabel('\gamma repair parameter');
    ylabel('Relative reconstruction error');
    title('\gamma vs Reconstruction Error for D_{LR}');
    legend('Reconstructed D_{LR}', 'Original Relative Error', 'Location', 'best');
    grid on;

    saveas(gcf, fullfile(plotDir, 'gamma_vs_relative_error_DLR.png'));
    savefig(gcf, fullfile(plotDir, 'gamma_vs_relative_error_DLR.fig'));


    % =========================================================
    % Plot 4: Gamma vs reconstruction error for D_LR2obf only
    % =========================================================
    figure('Color','w','Position',[100 100 900 600]);
    
    plot(gamma, summary.mean_relerr_DLR2obf, '-s', ...
        'LineWidth', 1.8, 'MarkerSize', 6);
    hold on;

    % Original unrepaired baseline at gamma = 0
    original_DLR_viol = summary.mean_viol_DLR(summary.gamma == 0);
    if ~isempty(original_DLR_viol)
        yline(original_DLR_viol, '--', ...
            sprintf('Original \\gamma=0: %.4f', original_DLR_viol), ...
            'LineWidth', 1.2);
    end

    hold off;
    xlabel('\gamma repair parameter');
    ylabel('Relative reconstruction error');
    title('\gamma vs Reconstruction Error for D_{LR2obf}');
    legend('Reconstructed D_{LR2obf}', 'Original Relative Error', 'Location', 'best');
    grid on;

    saveas(gcf, fullfile(plotDir, 'gamma_vs_relative_error_DLR2obf.png'));
    savefig(gcf, fullfile(plotDir, 'gamma_vs_relative_error_DLR2obf.fig'));

end

% function plot_gamma_violation(csv_file)
% 
%     results = readtable(csv_file);
% 
%     summaryFile = strrep(csv_file, '.csv', '_summary.csv');
%     summary = readtable(summaryFile);
% 
%     plotDir = fullfile('additional_experiment_results','results','plots');
%     if ~exist(plotDir, 'dir')
%         mkdir(plotDir);
%     end
% 
%     % ---------------------------------------------------------
%     % Plot 1: Gamma vs violation ratio
%     % ---------------------------------------------------------
%     figure('Color','w','Position',[100 100 900 600]);
% 
%     errorbar(summary.gamma, summary.mean_viol_DLR, summary.std_viol_DLR, ...
%         '-o', 'LineWidth', 1.6, 'MarkerSize', 6);
%     hold on;
%     errorbar(summary.gamma, summary.mean_viol_DLR2obf, summary.std_viol_DLR2obf, ...
%         '-s', 'LineWidth', 1.6, 'MarkerSize', 6);
%     hold off;
% 
%     xlabel('\gamma repair parameter');
%     ylabel('Violation ratio');
%     title('\gamma vs mDP Distance-Overestimation Violation Ratio');
%     legend('D_{LR}', 'D_{LR2obf}', 'Location', 'best');
%     grid on;
% 
%     saveas(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio.png'));
%     savefig(gcf, fullfile(plotDir, 'gamma_vs_violation_ratio.fig'));
% 
%     % ---------------------------------------------------------
%     % Plot 2: Gamma vs relative error
%     % ---------------------------------------------------------
%     figure('Color','w','Position',[100 100 900 600]);
% 
%     errorbar(summary.gamma, summary.mean_relerr_DLR, summary.std_relerr_DLR, ...
%         '-o', 'LineWidth', 1.6, 'MarkerSize', 6);
%     hold on;
%     errorbar(summary.gamma, summary.mean_relerr_DLR2obf, summary.std_relerr_DLR2obf, ...
%         '-s', 'LineWidth', 1.6, 'MarkerSize', 6);
%     hold off;
% 
%     xlabel('\gamma repair parameter');
%     ylabel('Relative error');
%     title('\gamma vs Reconstruction Error');
%     legend('D_{LR}', 'D_{LR2obf}', 'Location', 'best');
%     grid on;
% 
%     saveas(gcf, fullfile(plotDir, 'gamma_vs_relative_error.png'));
%     savefig(gcf, fullfile(plotDir, 'gamma_vs_relative_error.fig'));
% 
% end


function violRatio = violation_ratio(D_true, D_hat)

    if ~isequal(size(D_true), size(D_hat))
        error('D_true and D_hat must have the same size.');
    end

    numViolations = sum(D_hat(:) > D_true(:));
    totalEntries  = numel(D_true);

    violRatio = numViolations / totalEntries;

end


function err = relative_error(A, Ahat)

    err = norm(A - Ahat, 'fro') / max(1, norm(A, 'fro'));

end