% ======================================================================
% reorder_fit_rbfs.m
%
%  [best_pi, best_params, F1, F2, F3] = reorder_fit_rbfs( ...
%        A1, A2, A3, num_centres, sigma, lambda2, lambda3 )
%
%      • A1 : n×n  (symmetric)                      – original-vs-original
%      • A2 : n×m                                    – original-vs-obf
%      • A3 : n×m                                    – cost-coefficients
%
%  Each surface is modelled as             L
%        F(x,y) =  Σ  αℓ exp(‖[x,y]-cℓ‖² / (-2σ²))
%                                            ℓ=1
%    where {cℓ} are fixed radial centres on a coarse grid, σ is the common
%    bandwidth, and the αℓ are the learnable parameters we optimise.
%
%  OUTPUT
%  ▸ best_pi      : 1×n  index permutation (rows/cols of A1, rows of A2/A3)
%  ▸ best_params  : 1×(3·L) vector  [α₁…α_L , β₁…β_L , γ₁…γ_L]
%  ▸ F1,F2,F3     : fitted RBF surfaces after re-ordering is undone
% ======================================================================

function [best_pi, best_params, F1, F2, F3] = reorder_fit_rbfs( ...
            A1, A2, A3, num_centres, sigma, lambda2, lambda3)

    if nargin < 4 || isempty(num_centres), num_centres = 9;  end
    if nargin < 5 || isempty(sigma),       sigma        = max(size(A1))/3; end
    if nargin < 6 || isempty(lambda2),     lambda2      = 1;  end
    if nargin < 7 || isempty(lambda3),     lambda3      = 1;  end

    [n, m]  = size(A2);
    L       = num_centres;
    rng('default');

    % ------------------------------------------------------------
    % 0)  Fix a grid of RBF centres (same for all three surfaces)
    % ------------------------------------------------------------
    g        = ceil(sqrt(L));                    % square-ish grid
    cx       = linspace(1, n, g);
    cy       = linspace(1, max(n,m), g);
    [CX,CY]  = meshgrid(cx,cy);
    centres  = [CX(:), CY(:)];                   % L×2 matrix (rows are centres)
    centres  = centres(1:L,:);                  % keep first L only

    % ------------------------------------------------------------
    % Helper to build RBF surface --------------------------------
    % ------------------------------------------------------------
    function Z = eval_rbf2d(weights, rows, cols)
        [X,Y] = meshgrid(1:cols,1:rows);        % X = col-index, Y = row-index
        Z     = zeros(rows, cols);
        for k = 1:L
            dx2   = (X - centres(k,1)).^2 + (Y - centres(k,2)).^2;
            Z     = Z + weights(k) * exp(-dx2/(2*sigma^2));
        end
    end

    % ------------------------------------------------------------
    % 1) Start with identity perm + random weights --------------
    % ------------------------------------------------------------
    pi_curr     = 1:n;
    params_curr = randn(1, 3*L);                % α,β,γ  stacked
    opts        = optimset('Display','off','MaxFunEvals',2e3);

    params_curr = fminsearch(@(p) lossFun(p,pi_curr), params_curr, opts);
    loss_curr   = lossFun(params_curr,pi_curr);

    % ------------------------------------------------------------
    % 2) Greedy swap search  +  re-fit weights -------------------
    % ------------------------------------------------------------
    MAX_IT = 10;
    for it = 1:MAX_IT
        improved = false;
        for i = 1:n
            for j = i+1:n
                pi_try        = pi_curr;
                pi_try([i j]) = pi_try([j i]);
                loss_try      = lossFun(params_curr,pi_try);
                if loss_try < loss_curr
                    pi_curr   = pi_try;
                    loss_curr = loss_try;
                    improved  = true;
                end
            end
        end
        params_curr = fminsearch(@(p) lossFun(p,pi_curr), params_curr, opts);
        loss_curr   = lossFun(params_curr,pi_curr);
        if ~improved, break; end
    end

    best_pi     = pi_curr;
    best_params = params_curr;

    % Build final fitted surfaces (undo permutation so they align with A1/A2/A3)
    [F1p,F2p,F3p]   = build_surfaces(best_params);
    inv_pi           = zeros(1,n);  inv_pi(best_pi) = 1:n;
    F1 = F1p(inv_pi,inv_pi);
    F2 = F2p(inv_pi,:);
    F3 = F3p(inv_pi,:);

    fprintf('Best permutation: [%s]\n', num2str(best_pi));
    fprintf('Best RBF weights (L = %d):\n', L);
    disp(reshape(best_params,L,3)');
    fprintf('Minimum L1 loss: %.4f\n', loss_curr);

    % --------------- nested helpers ---------------------------------
    function [F1s,F2s,F3s] = build_surfaces(p)
        a  = p(1:L);               % weights for A1
        b  = p(L+1:2*L);           % weights for A2
        c  = p(2*L+1:end);         % weights for A3
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
end
