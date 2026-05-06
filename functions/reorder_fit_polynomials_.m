function [best_pi, best_coeffs, F1, F2, F3, stats] = reorder_fit_polynomials_(A1, A2, A3, deg, lambda2, lambda3)
%REORDER_FIT_POLYNOMIALS_ Alternating scheme (perm swaps + coeff refit) with instrumentation
%
% Adds (same as gaussian version):
%   - per-outer-iteration timing (swap search vs fminsearch)
%   - loss history after swap+refit
%   - best loss during swap phase (swap-only best)
%   - refit-only diagnostic loss (no swaps)
%   - swap eval/accept counts
%   - max_iter cap
%
% stats fields:
%   .outer_iters, .converged, .stop_reason
%   .loss_hist (iter 0..K)
%   .loss_swap_best (iter 1..K)
%   .loss_refit_only (iter 1..K)
%   .time_swap, .time_fit, .time_fit_init, .time_refit_only, .time_total
%   .swap_evals, .swap_accepts

    if nargin < 4 || isempty(deg),      deg      = 3;   end
    if nargin < 5 || isempty(lambda2),  lambda2  = 1;   end
    if nargin < 6 || isempty(lambda3),  lambda3  = 1;   end

    t_total = tic;

    [n, m] = size(A2);
    L = (deg + 1)*(deg + 2)/2;      % # monomials per surface
    pi_current = 1:n;

    % ---- Hyperparameters ----
    max_iter = 10;
    tol_rel  = 0;
    tol_abs  = 0;
    max_fun_evals = 5000;

    rng('default');
    coeffs_current = randn(1, 3*L);
    opts = optimset('Display','off','MaxFunEvals',max_fun_evals);

    % ---- Initial coefficient fit ----
    t_fit0 = tic;
    coeffs_current = fminsearch(@(c) loss_fun(A1,A2,A3,pi_current,c,deg,lambda2,lambda3), ...
                                coeffs_current, opts);
    time_fit_init = toc(t_fit0);

    loss_current = loss_fun(A1,A2,A3,pi_current,coeffs_current,deg,lambda2,lambda3);

    % ---- Preallocate stats ----
    loss_hist       = nan(max_iter+1,1);   % loss after swap+refit (iter 0..K)
    loss_swap_best  = nan(max_iter,1);     % best loss reached during swap phase (iter 1..K)
    loss_refit_only = nan(max_iter,1);     % refit-only diagnostic (iter 1..K)

    time_swap       = zeros(max_iter,1);
    time_fit        = zeros(max_iter,1);
    time_refit_only = zeros(max_iter,1);

    swap_evals      = zeros(max_iter,1);
    swap_accepts    = zeros(max_iter,1);

    loss_hist(1) = loss_current;

    converged   = false;
    stop_reason = "max_iter_reached";

    % ---- Alternating outer loop ----
    for iter = 1:max_iter

        % ===== (0) Refit-only diagnostic (no swaps) =====
        pi_before = pi_current;
        c_before  = coeffs_current;

        t_ref = tic;
        c_refit_only = fminsearch(@(c) loss_fun(A1,A2,A3,pi_before,c,deg,lambda2,lambda3), ...
                                  c_before, opts);
        time_refit_only(iter) = toc(t_ref);
        loss_refit_only(iter) = loss_fun(A1,A2,A3,pi_before,c_refit_only,deg,lambda2,lambda3);

        improved = false;

        % ===== (1) Permutation improvement by greedy swaps =====
        t_swap_iter = tic;

        evals_this   = 0;
        accepts_this = 0;
        best_loss_in_swaps = loss_current;

        for ii = 1:n
            for jj = ii+1:n
                evals_this = evals_this + 1;

                pi_trial = pi_current;
                pi_trial([ii jj]) = pi_trial([jj ii]);

                loss_trial = loss_fun(A1,A2,A3,pi_trial,coeffs_current,deg,lambda2,lambda3);

                if loss_trial < loss_current
                    pi_current   = pi_trial;
                    loss_current = loss_trial;
                    improved     = true;
                    accepts_this = accepts_this + 1;
                end

                if loss_current < best_loss_in_swaps
                    best_loss_in_swaps = loss_current;
                end
            end
        end

        time_swap(iter)      = toc(t_swap_iter);
        swap_evals(iter)     = evals_this;
        swap_accepts(iter)   = accepts_this;
        loss_swap_best(iter) = best_loss_in_swaps;

        % ===== (2) Refit coeffs given new permutation =====
        t_fit_iter = tic;
        coeffs_current = fminsearch(@(c) loss_fun(A1,A2,A3,pi_current,c,deg,lambda2,lambda3), ...
                                    coeffs_current, opts);
        time_fit(iter) = toc(t_fit_iter);

        loss_new = loss_fun(A1,A2,A3,pi_current,coeffs_current,deg,lambda2,lambda3);

        loss_current = loss_new;            % keep consistent
        loss_hist(iter+1) = loss_current;   % log after swap+refit

        % ===== stopping =====
        if ~improved
            converged = true;
            stop_reason = "no_improving_swap";
            break;
        end
    end

    outer_iters = find(~isnan(loss_hist), 1, 'last') - 1;

    % Trim arrays
    loss_hist_trim  = loss_hist(1:outer_iters+1);
    loss_swap_best  = loss_swap_best(1:outer_iters);
    loss_refit_only = loss_refit_only(1:outer_iters);

    time_swap       = time_swap(1:outer_iters);
    time_fit        = time_fit(1:outer_iters);
    time_refit_only = time_refit_only(1:outer_iters);

    swap_evals      = swap_evals(1:outer_iters);
    swap_accepts    = swap_accepts(1:outer_iters);

    % Outputs
    best_pi     = pi_current;
    best_coeffs = coeffs_current;
    [F1, F2, F3] = generate_polynomials(n, m, best_coeffs, deg);

    % Stats
    stats = struct();
    stats.outer_iters   = outer_iters;
    stats.converged     = converged;
    stats.stop_reason   = char(stop_reason);

    stats.loss_hist       = loss_hist_trim;
    stats.loss_swap_best  = loss_swap_best;
    stats.loss_refit_only = loss_refit_only;

    stats.swap_evals    = swap_evals;
    stats.swap_accepts  = swap_accepts;

    stats.time_swap       = time_swap;
    stats.time_fit        = time_fit;
    stats.time_fit_init   = time_fit_init;
    stats.time_refit_only = time_refit_only;

    stats.time_total    = toc(t_total);

    fprintf('Final loss: %.6f | outer iters: %d | stop: %s | total time: %.3fs\n', ...
            loss_hist_trim(end), outer_iters, stats.stop_reason, stats.time_total);
end

% ================= helpers =================
function loss = loss_fun(A1,A2,A3,pi,coeffs,deg,lambda2,lambda3)
    [n,m] = size(A2);
    [F1,F2,F3] = generate_polynomials(n,m,coeffs,deg);

    A1p = A1(pi,pi);
    A2p = A2(pi,:);
    A3p = A3(pi,:);

    loss1 = sum(abs(A1p(:) - F1(:)));
    loss2 = sum(abs(A2p(:) - F2(:)));
    loss3 = sum(abs(A3p(:) - F3(:)));

    loss = loss1 + lambda2*loss2 + lambda3*loss3;
end

function [F1,F2,F3] = generate_polynomials(n,m,coeffs,deg)
    L  = (deg + 1)*(deg + 2)/2;
    c1 = coeffs(1:L);
    c2 = coeffs(L+1:2*L);
    c3 = coeffs(2*L+1:end);

    F1 = eval_poly2d(c1, n, n, deg);
    F2 = eval_poly2d(c2, n, m, deg);
    F3 = eval_poly2d(c3, n, m, deg);
end

function Z = eval_poly2d(c, rows, cols, deg)
    [X,Y] = meshgrid(1:cols, 1:rows);
    Z = zeros(rows, cols);

    k = 0;
    for p = 0:deg
        for q = 0:(deg - p)
            k = k + 1;
            Z = Z + c(k) * (X.^p) .* (Y.^q);
        end
    end
end