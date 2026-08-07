function [best_pi, best_params, F1, F2, F3, stats] = reorder_fit_gaussians(A1, A2, A3, lambda2, lambda3)
%REORDER_FIT_GAUSSIANS Alternating scheme with instrumentation + convergence logging
%
% Adds:
%   - per-outer-iteration timing (swap search vs fminsearch)
%   - loss history
%   - swap eval/accept counts
%   - tolerance-based stopping + max_iter cap
%
% Output:
%   stats struct with fields:
%     .outer_iters, .converged, .stop_reason
%     .loss_hist
%     .swap_evals, .swap_accepts
%     .time_swap, .time_fit, .time_total

    t_total = tic;

    [n, m] = size(A2);
    pi_current = 1:n; % Start with identity permutation

    % ---- Hyperparameters for convergence study ----
    max_iter = 20;            % cap (report how often reached)
    tol_rel  = 0;             % relative improvement tolerance
    tol_abs  = 0;             % optional absolute tolerance (set >0 if you want)
    max_fun_evals = 1000;     % for fminsearch

    % Random initial Gaussian parameters [cx, cy, sx, sy, A, b] for each of 3 components
    params_current = [
        n/2, n/2, n/4, n/4, 1.0, 0.0, ... % F1
        n/2, m/2, n/4, m/4, 1.0, 0.0, ... % F2
        n/2, m/2, n/4, m/4, 1.0, 0.0      % F3
    ];

    % fminsearch options
    options = optimset('Display','off', 'MaxFunEvals', max_fun_evals);

    % ---- Initial parameter fit ----
    t_fit0 = tic;
    params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), ...
                                params_current, options);
    time_fit_init = toc(t_fit0);

    loss_current = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);

    % ---- Preallocate stats ----
    loss_hist       = nan(max_iter+1, 1);   % loss after refit (outer iteration)
    loss_swap_best  = nan(max_iter, 1);     % best loss achieved during swap search (optional)
    time_swap       = zeros(max_iter, 1);
    time_fit        = zeros(max_iter, 1);
    swap_evals      = zeros(max_iter, 1);
    swap_accepts    = zeros(max_iter, 1);
    loss_refit_only = nan(max_iter, 1);     % refit-only diagnostic
    time_refit_only = zeros(max_iter, 1);   % optional timing


    loss_hist(1) = loss_current;

    converged  = false;
    stop_reason = "max_iter_reached";

    % ---- Alternating outer loop ----
    for iter = 1:max_iter
        % ---- Refit-only diagnostic (no swaps) ----
        pi_before = pi_current;
        p_before  = params_current;
        
        t_refit_only = tic;
        p_refit_only = fminsearch(@(p) compute_loss(A1, A2, A3, pi_before, p, lambda2, lambda3), ...
                                  p_before, options);
        time_refit_only(iter) = toc(t_refit_only);

        loss_refit_only(iter) = compute_loss(A1, A2, A3, pi_before, p_refit_only, lambda2, lambda3);
        %---------------------

        improved = false;

        % ===== (1) Permutation improvement by greedy swaps =====
        t_swap_iter = tic;
        accepts_this_iter = 0;
        evals_this_iter   = 0;
        best_loss_in_swaps = loss_current;          % track best loss reached in swap phase

        for i = 1:n
            for j = i+1:n
                evals_this_iter = evals_this_iter + 1;
    
                pi_trial = pi_current;
                pi_trial([i j]) = pi_trial([j i]);  % swap
    
                loss_trial = compute_loss(A1, A2, A3, pi_trial, params_current, lambda2, lambda3);
    
                if loss_trial < loss_current
                    pi_current   = pi_trial;
                    loss_current = loss_trial;      % <-- this is the "loss_current during swaps"
                    improved     = true;
                    accepts_this_iter = accepts_this_iter + 1;
                end
    
                if loss_current < best_loss_in_swaps
                    best_loss_in_swaps = loss_current;
                end
            end
        end
    
        time_swap(iter)    = toc(t_swap_iter);
        swap_evals(iter)   = evals_this_iter;
        swap_accepts(iter) = accepts_this_iter;
        loss_swap_best(iter) = best_loss_in_swaps;   % optional
    
        % ===== (2) Re-optimize Gaussian parameters given new permutation =====
        t_fit_iter = tic;
        params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), ...
                                    params_current, options);
        time_fit(iter) = toc(t_fit_iter);
    
        % Loss after refit
        loss_new = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);
    
        loss_current = loss_new;          % <-- IMPORTANT: keep loss_current consistent
        loss_hist(iter+1) = loss_current; % <-- log per-iteration loss_current
    
        % ===== Stopping criteria =====
        if ~improved
            converged = true;
            stop_reason = "no_improving_swap";
            break;
        end

        % if (rel_impr < tol_rel) || (tol_abs > 0 && abs_impr < tol_abs)
        %     converged = true;
        %     stop_reason = "tolerance_reached";
        %     loss_current = loss_new;
        %     break;
        % end

        % loss_current = loss_new;
    end

    outer_iters = find(~isnan(loss_hist), 1, 'last') - 1; % number of completed outer iters

    % Trim arrays to actual iterations
    loss_hist_trim  = loss_hist(1:outer_iters+1);
    loss_swap_best  = loss_swap_best(1:outer_iters);
    time_swap       = time_swap(1:outer_iters);
    time_fit        = time_fit(1:outer_iters);
    swap_evals      = swap_evals(1:outer_iters);
    swap_accepts    = swap_accepts(1:outer_iters);
    loss_refit_only = loss_refit_only(1:outer_iters);
    time_refit_only = time_refit_only(1:outer_iters);


    % Final outputs
    best_pi     = pi_current;
    best_params = params_current;

    [F1, F2, F3] = generate_gaussians(n, m, best_params);

    % Stats package
    stats = struct();
    stats.outer_iters   = outer_iters;
    stats.converged     = converged;
    stats.stop_reason   = char(stop_reason);
    stats.loss_hist     = loss_hist_trim;         % loss_current after each refit
    stats.swap_evals    = swap_evals;
    stats.swap_accepts  = swap_accepts;
    stats.time_swap     = time_swap;
    stats.time_fit      = time_fit;
    stats.time_fit_init = time_fit_init;
    stats.time_total    = toc(t_total);       
    stats.loss_swap_best  = loss_swap_best;        % optional: best loss during swap phase
    stats.loss_refit_only = loss_refit_only;
    stats.time_refit_only = time_refit_only;       % optional



    % Optional prints (turn off if too verbose)
    fprintf('Best permutation: [%s]\n', num2str(best_pi));
    fprintf('Best Gaussian params (6 per component):\n');
    disp(reshape(best_params, 6, 3)');
    fprintf('Final loss: %.6f | outer iters: %d | stop: %s | total time: %.3fs\n', ...
            loss_hist_trim(end), outer_iters, stats.stop_reason, stats.time_total);
end


% ========================= Helper functions (unchanged logic) =========================
function loss = compute_loss(A1, A2, A3, pi, params, lambda2, lambda3)
    [n, m] = size(A2);
    [F1, F2, F3] = generate_gaussians(n, m, params);

    A1p = A1(pi, pi);
    A2p = A2(pi, :);
    A3p = A3(pi, :);

    loss1 = sum(abs(A1p(:) - F1(:)));
    loss2 = sum(abs(A2p(:) - F2(:)));
    loss3 = sum(abs(A3p(:) - F3(:)));

    loss = loss1 + lambda2 * loss2 + lambda3 * loss3;
end

function [F1, F2, F3] = generate_gaussians(n, m, params)
    [I1, J1] = meshgrid(1:n, 1:n);
    [I2, J2] = meshgrid(1:n, 1:m);

    p1 = params(1:6);
    p2 = params(7:12);
    p3 = params(13:18);

    F1 = gaussian2d(I1, J1, p1(1), p1(2), p1(3), p1(4), p1(5), p1(6));
    F2 = gaussian2d(I2, J2, p2(1), p2(2), p2(3), p2(4), p2(5), p2(6));
    F3 = gaussian2d(I2, J2, p3(1), p3(2), p3(3), p3(4), p3(5), p3(6));
end

function val = gaussian2d(x, y, cx, cy, sx, sy, A, b)
    exponent = -((x - cx).^2 ./ (2 * sx^2) + (y - cy).^2 ./ (2 * sy^2));
    val = A * exp(exponent) + b;
end
