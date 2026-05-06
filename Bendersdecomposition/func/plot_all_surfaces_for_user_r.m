function plot_all_surfaces_for_user_r(u, outdir)
%PLOT_ALL_SURFACES_FOR_USER Make surface plots for original vs. recovered matrices.
% u      : a User object from your script (e.g., user(m))
% outdir : folder to save PNGs (created if missing)

    if nargin < 2 || isempty(outdir), outdir = "london_fig_surfaces_6000/fig_surfaces_r"; end
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    % 1) distance_matrix_LR
    plot_matrix_pair_surf( ...
        u.distance_matrix_LR, ...
        u.distance_matrix_LR_recovered_r, ...
        'distance\_matrix\_LR (original)', ...
        'distance\_matrix\_LR\_recovered_r', ...
        fullfile(outdir, sprintf('surf_LR_user_r%02d.png', u.loc_ID)) );

    % 2) distance_matrix_LR2obf
    plot_matrix_pair_surf( ...
        u.distance_matrix_LR2obf, ...
        u.distance_matrix_LR2obf_recovered_r, ...
        'distance\_matrix\_LR2obf (original)', ...
        'distance\_matrix\_LR2obf\_recovered_r', ...
        fullfile(outdir, sprintf('surf_LR2obf_user_r%02d.png', u.loc_ID)) );

    % 3) cost_matrix_RL
    plot_matrix_pair_surf( ...
        u.cost_matrix_RL, ...
        u.cost_matrix_RL_recovered_r, ...
        'cost\_matrix\_RL (original)', ...
        'cost\_matrix\_RL\_recovered_r', ...
        fullfile(outdir, sprintf('surf_cost_RL_user_r%02d.png', u.loc_ID)) );
end

function plot_matrix_pair_surf(A, B, titleA, titleB, savepath)
%PLOT_MATRIX_PAIR_SURF Surface plots for a pair of matrices with matched color/height scales.
% Handles empty/NaN gracefully and adds rel error / violation in the supertitle if helpers exist.

    if nargin < 5, savepath = []; end

    if isempty(A)
        warning('Original matrix is empty; skipping plot for %s.', titleA);
        return
    end
    if isempty(B)
        warning('Recovered matrix is empty; skipping plot for %s.', titleB);
        return
    end
    if ~isequal(size(A), size(B))
        warning('Size mismatch for %s vs %s; skipping.', titleA, titleB);
        return
    end

    % Common limits across the pair (ignore NaNs)
    lo = min([min(A,[],'all','omitnan'), min(B,[],'all','omitnan')]);
    hi = max([max(A,[],'all','omitnan'), max(B,[],'all','omitnan')]);
    if ~isfinite(lo) || ~isfinite(hi) || lo==hi
        lo = 0; hi = 1; % fallback
    end

    % Build grids for surf
    [nRows, nCols] = size(A);
    [X, Y] = meshgrid(1:nCols, 1:nRows);

    % Compute metrics if available
    hasRel = exist('relative_error','file') == 2;
    hasVio = exist('violation_ratio','file') == 2;
    msg = "";
    if hasRel
        rel = relative_error(A, B);
        msg = msg + sprintf('  RelErr = %.4g', rel);
    end
    if hasVio
        vio = violation_ratio(A, B);
        msg = msg + sprintf('  |  ViolRatio = %.4g', vio);
    end

    % Plot
    f = figure('Color','w','Position',[100 100 1400 520]);
    tl = tiledlayout(1,2, 'TileSpacing','compact','Padding','compact');

    % Left: Original
    nexttile;
    s1 = surf(X, Y, A);
    shading interp; view(45,30);
    colormap parula; % uses current default colormap
    cb1 = colorbar; cb1.Label.String = 'Value';
    caxis([lo, hi]); zlim([lo, hi]);
    xlabel('Column'); ylabel('Row'); zlabel('Value');
    title(titleA, 'Interpreter','none');

    % Right: Recovered
    nexttile;
    s2 = surf(X, Y, B);
    shading interp; view(45,30);
    colormap parula;
    cb2 = colorbar; cb2.Label.String = 'Value';
    caxis([lo, hi]); zlim([lo, hi]);
    xlabel('Column'); ylabel('Row'); zlabel('Value');
    title(titleB, 'Interpreter','none');

    % Keep axes/lighting consistent
    linkprop([s1.Parent, s2.Parent], {'CLim','ZLim','View'});

    % Suptitle with metrics
    if strlength(msg) > 0
        title(tl, sprintf('Surface comparison%s', msg));
    else
        title(tl, 'Surface comparison');
    end

    % Optional: also show a 2D difference heatmap in a separate figure
    try
        f2 = figure('Color','w','Position',[100 100 700 500]);
        imagesc(A - B);
        axis image; colorbar; xlabel('Column'); ylabel('Row');
        title('Difference heatmap: A - B');
        if ~isempty(savepath)
            [p,n,~] = fileparts(savepath);
            saveas(f2, fullfile(p, [n '_diff.png']));
        end
    catch
        % non-fatal
    end

    % Save
    if ~isempty(savepath)
        exportgraphics(f, savepath, 'Resolution', 200);
    end
end
