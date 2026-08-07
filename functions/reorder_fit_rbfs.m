function [best_pi, best_params, F1, F2, F3, stats] = reorder_fit_rbfs( ...
            A1, A2, A3, num_centres, sigma, lambda2, lambda3)

    t_total = tic;

    if nargin < 4 || isempty(num_centres), num_centres = 9;  end
    if nargin < 5 || isempty(sigma),       sigma        = max(size(A1))/3; end
    if nargin < 6 || isempty(lambda2),     lambda2      = 1;  end
    if nargin < 7 || isempty(lambda3),     lambda3      = 1;  end

    [n, m]  = size(A2);
    L       = num_centres;
    rng('default');

    % ------------------------------------------------------------
    % 0) Fix RBF centres on a grid
    % ------------------------------------------------------------
    g        = ceil(sqrt(L));
    cx       = linspace(1, n, g);
    cy       = linspace(1, max(n,m), g);
    [CX,CY]  = meshgrid(cx,cy);
    centres  = [CX(:), CY(:)];
    centres  = centres(1:L,:);

    % RBF surface eval
    function Z = eval_rbf2d(weights, rows, cols)
        [X,Y] = meshgrid(1:cols,1:rows);
        Z     = zeros(rows, cols);
        for k = 1:L
            dx2 = (X - centres(k,1)).^2 + (Y - centres(k,2)).^2;
            Z   = Z + weights(k) * exp(-dx2/(2*sigma^2));
        end
    end

    % build 3 surfaces from stacked weights
    function [F1s,F2s,F3s] = build_surfaces(p)
        a  = p(1:L);
        b  = p(L+1:2*L);
        c  = p(2*L+1:end);
        F1s = eval_rbf2d(a, n, n);
        F2s = eval_rbf2d(b, n, m);
        F3s = eval_rbf2d(c, n, m);
    end

    function loss = lossFun(p, pi)
        [F1s,F2s,F3s] = build_surfaces(p);
        A1p = A1(pi,pi);
        A2p = A2(pi,:);
        A3p = A3(pi,:);
        loss = sum(abs(A1p(:)-F1s(:))) + ...
               lambda2*sum(abs(A2p(:)-F2s(:))) + ...
               lambda3*sum(abs(A3p(:)-F3s(:)));
    end

    % ------------------------------------------------------------
    % 1) Init
    % ------------------------------------------------------------
    pi_curr     = 1:n;
    params_curr = randn(1, 3*L);

    max_iter = 20;
    max_fun_evals = 2000;
    opts = optimset('Display','off','MaxFunEvals',max_fun_evals);

    % initial fit
    t_fit0 = tic;
    params_curr = fminsearch(@(p) lossFun(p,pi_curr), params_curr, opts);
    time_fit_init = toc(t_fit0);

    loss_curr = lossFun(params_curr,pi_curr);

    % ------------------------------------------------------------
    % 2) Stats buffers (MATCH GAUSSIAN)
    % ------------------------------------------------------------
    loss_hist       = nan(max_iter+1,1);   % after swap+refit (iter 0..K)
    loss_swap_best  = nan(max_iter,1);     % best loss reached during swap phase (iter 1..K)
    loss_refit_only = nan(max_iter,1);     % refit-only diagnostic (iter 1..K)

    time_swap       = zeros(max_iter,1);
    time_fit        = zeros(max_iter,1);
    time_refit_only = zeros(max_iter,1);

    swap_evals      = zeros(max_iter,1);
    swap_accepts    = zeros(max_iter,1);

    loss_hist(1) = loss_curr;

    converged   = false;
    stop_reason = "max_iter_reached";

    % ------------------------------------------------------------
    % 3) Alternating: (0) refit-only diag, (1) swap, (2) refit
    % ------------------------------------------------------------
    for it = 1:max_iter

        % ===== (0) Refit-only diagnostic (NO swaps) =====
        pi_before = pi_curr;
        p_before  = params_curr;

        t_ref = tic;
        p_refit_only = fminsearch(@(p) lossFun(p,pi_before), p_before, opts);
        time_refit_only(it) = toc(t_ref);
        loss_refit_only(it) = lossFun(p_refit_only, pi_before);

        improved = false;

        % ===== (1) swap search =====
        t_swap_it = tic;
        evals = 0; accepts = 0;
        best_loss_in_swaps = loss_curr;

        for i = 1:n
            for j = i+1:n
                evals = evals + 1;

                pi_try        = pi_curr;
                pi_try([i j]) = pi_try([j i]);

                loss_try = lossFun(params_curr,pi_try);
                if loss_try < loss_curr
                    pi_curr   = pi_try;
                    loss_curr = loss_try;      % <-- "loss during swaps"
                    improved  = true;
                    accepts   = accepts + 1;
                end

                if loss_curr < best_loss_in_swaps
                    best_loss_in_swaps = loss_curr;
                end
            end
        end

        time_swap(it)      = toc(t_swap_it);
        swap_evals(it)     = evals;
        swap_accepts(it)   = accepts;
        loss_swap_best(it) = best_loss_in_swaps;

        % ===== (2) refit weights after swaps =====
        t_fit_it = tic;
        params_curr = fminsearch(@(p) lossFun(p,pi_curr), params_curr, opts);
        time_fit(it) = toc(t_fit_it);

        loss_new = lossFun(params_curr,pi_curr);
        loss_curr = loss_new;
        loss_hist(it+1) = loss_curr;

        % stopping
        if ~improved
            converged   = true;
            stop_reason = "no_improving_swap";
            break;
        end
    end

    outer_iters = find(~isnan(loss_hist),1,'last') - 1;

    % trim
    loss_hist       = loss_hist(1:outer_iters+1);
    loss_swap_best  = loss_swap_best(1:outer_iters);
    loss_refit_only = loss_refit_only(1:outer_iters);

    time_swap       = time_swap(1:outer_iters);
    time_fit        = time_fit(1:outer_iters);
    time_refit_only = time_refit_only(1:outer_iters);

    swap_evals      = swap_evals(1:outer_iters);
    swap_accepts    = swap_accepts(1:outer_iters);

    % ------------------------------------------------------------
    % 4) Final outputs (undo permutation)  [SAME AS YOURS]
    % ------------------------------------------------------------
    best_pi     = pi_curr;
    best_params = params_curr;

    [F1p,F2p,F3p] = build_surfaces(best_params);

    inv_pi = zeros(1,n);
    inv_pi(best_pi) = 1:n;

    F1 = F1p(inv_pi,inv_pi);
    F2 = F2p(inv_pi,:);
    F3 = F3p(inv_pi,:);

    % ------------------------------------------------------------
    % 5) stats struct (MATCH GAUSSIAN)
    % ------------------------------------------------------------
    stats = struct();
    stats.outer_iters     = outer_iters;
    stats.converged       = converged;
    stats.stop_reason     = char(stop_reason);

    stats.loss_hist       = loss_hist;
    stats.loss_swap_best  = loss_swap_best;
    stats.loss_refit_only = loss_refit_only;

    stats.swap_evals      = swap_evals;
    stats.swap_accepts    = swap_accepts;

    stats.time_swap       = time_swap;
    stats.time_fit        = time_fit;
    stats.time_fit_init   = time_fit_init;
    stats.time_refit_only = time_refit_only;

    stats.time_total      = toc(t_total);

    fprintf('Final loss: %.6f | outer iters: %d | stop: %s | total time: %.3fs\n', ...
        loss_hist(end), outer_iters, stats.stop_reason, stats.time_total);
end