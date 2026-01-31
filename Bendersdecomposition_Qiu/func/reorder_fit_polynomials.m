% ========================================================================
% reorder_fit_polynomials.m
%  ▸ Re‑implements your Gaussian routine with *2‑D polynomial* surfaces.
%  ▸ Keeps the same signature/outputs so you can drop‑replace it.
%
%  [best_pi, best_coeffs, F1, F2, F3] = reorder_fit_polynomials( ...
%          A1, A2, A3, deg, lambda2, lambda3 )
%
%     • A1 : n×n  (symmetric)
%     • A2 : n×m
%     • A3 : n×m
%     • deg: polynomial degree (default = 3 ⇒ 10 coeffs / surface)
%     • λ₂,λ₃: weighting terms for A2,A3 in the L1 loss
%
%  best_coeffs is a row vector containing the concatenated coefficients
%  for the three fitted polynomials:  [c₁ … c_L , d₁ … d_L , e₁ … e_L]
%  where L = (deg+1)(deg+2)/2.
%
%  You can add noise exactly as before:
%       noisy_coeffs = add_noise_to_distance_matrix(best_coeffs, EPSILON);
% ========================================================================

function [best_pi, best_coeffs, F1, F2, F3] = reorder_fit_polynomials(A1, A2, A3, deg, lambda2, lambda3)

    if nargin < 4 || isempty(deg),      deg      = 3;   end
    if nargin < 5 || isempty(lambda2),  lambda2  = 1;   end
    if nargin < 6 || isempty(lambda3),  lambda3  = 1;   end

    [n, m]   = size(A2);
    L        = (deg + 1)*(deg + 2)/2;           % # monomials
    rng('default');

    %------------------------------------------------------------------
    % 1)  Initialise permutation and polynomial coefficients
    %------------------------------------------------------------------
    pi_current      = 1:n;                      % identity perm
    coeffs_current  = randn(1, 3*L);            % random start
    opts            = optimset('Display','off','MaxFunEvals',5e3);

    % optimise coeffs w.r.t. current permutation
    coeffs_current  = fminsearch(@(c) loss_fun(A1,A2,A3,pi_current,c,...
                                               deg,lambda2,lambda3), ...
                                 coeffs_current, opts);

    loss_current    = loss_fun(A1,A2,A3,pi_current,coeffs_current,...
                               deg,lambda2,lambda3);

    %------------------------------------------------------------------
    % 2)  Alternating optimisation: greedy swaps + re‑fit coefficients
    %------------------------------------------------------------------
    MAX_ITER = 10;
    for iter = 1:MAX_ITER
        improved = false;

        for i = 1:n
            for j = i+1:n
                pi_trial         = pi_current;
                pi_trial([i j])  = pi_trial([j i]);

                loss_trial = loss_fun(A1,A2,A3,pi_trial,coeffs_current,...
                                      deg,lambda2,lambda3);

                if loss_trial < loss_current
                    pi_current   = pi_trial;
                    loss_current = loss_trial;
                    improved     = true;
                end
            end
        end

        % re‑optimise polynomial coefficients for the new permutation
        coeffs_current = fminsearch(@(c) loss_fun(A1,A2,A3,pi_current,c,...
                                                  deg,lambda2,lambda3), ...
                                    coeffs_current, opts);
        loss_current   = loss_fun(A1,A2,A3,pi_current,coeffs_current,...
                                  deg,lambda2,lambda3);

        if ~improved,  break;  end
    end

    %------------------------------------------------------------------
    % 3)  Return best permutation, coeffs, and reconstructed surfaces
    %------------------------------------------------------------------
    best_pi      = pi_current;
    best_coeffs  = coeffs_current;

    [F1, F2, F3] = generate_polynomials(n, m, best_coeffs, deg);

    fprintf('Best permutation: [%s]\n', num2str(best_pi));
    fprintf('Best polynomial coeffs (degree %d):\n', deg);
    disp(reshape(best_coeffs, L, 3)');
    fprintf('Minimum L1 loss: %.4f\n', loss_current);
end
% ========================================================================
% Loss  (L1)  ------------------------------------------------------------
function loss = loss_fun(A1, A2, A3, pi, coeffs, deg, lambda2, lambda3)
    [n, m]           = size(A2);
    [F1, F2, F3]     = generate_polynomials(n, m, coeffs, deg);

    A1p = A1(pi, pi);
    A2p = A2(pi, :);
    A3p = A3(pi, :);

    loss1 = sum(abs(A1p(:) - F1(:)));
    loss2 = sum(abs(A2p(:) - F2(:)));
    loss3 = sum(abs(A3p(:) - F3(:)));

    % loss  = loss1 + lambda2*loss2 + lambda3*loss3;
    lambdaR = 1e-3;  % tune via validation
    loss = loss1 + lambda2*loss2 + lambda3*loss3 + lambdaR * sum(coeffs(:).^2);
end
% ========================================================================
% Build polynomial surfaces  ---------------------------------------------
function [F1, F2, F3] = generate_polynomials(n, m, coeffs, deg)
    L   = (deg + 1)*(deg + 2)/2;
    c1  = coeffs(1:L);
    c2  = coeffs(L+1:2*L);
    c3  = coeffs(2*L+1:end);

    F1  = eval_poly2d(c1, n, n, deg);
    F2  = eval_poly2d(c2, n, m, deg);
    F3  = eval_poly2d(c3, n, m, deg);
end
% ------------------------------------------------------------------------
% function Z = eval_poly2d(c, rows, cols, deg)
%     [X, Y] = meshgrid(linspace(0,1,cols), linspace(0,1,rows));  
%     Z      = zeros(rows, cols);
% 
%     k = 0;
%     for p = 0:deg
%         for q = 0:(deg - p)
%             k   = k + 1;
%             Z   = Z + c(k) * (X.^p) .* (Y.^q);
%         end
%     end
% end


function Z = eval_poly2d(c, rows, cols, deg)
    [X, Y] = meshgrid(linspace(0,1,cols), linspace(0,1,rows));
    Z = eval_poly2d_coeffs(c, X, Y, deg);
end

function Z = eval_poly2d_coeffs(c, X, Y, deg)
% Legendre product basis on [0,1]:
%   P_0(t)=1, P_1(t)=2t-1, P_2(t)=6t^2-6t+1, ...
% Basis order matches your (p,q) loops: p=0..deg, q=0..deg-p
    Z = zeros(size(X));

    % Precompute Legendre up to deg for X and Y
    Px = legendre_01(X, deg);   % cell{p+1}, each size(X)
    Py = legendre_01(Y, deg);

    k = 0;
    for p = 0:deg
        for q = 0:(deg - p)
            k = k + 1;
            Z = Z + c(k) .* Px{p+1} .* Py{q+1};
        end
    end
end

function P = legendre_01(T, deg)
% Returns cell array P{r+1} = P_r( T ) with P_r Legendre on [0,1].
% Recurrence: in variable u=2t-1, standard Legendre on [-1,1].
    u = 2*T - 1;
    P = cell(deg+1,1);
    P{1} = ones(size(T));         % P0(u)=1
    if deg==0, return; end
    P{2} = u;                      % P1(u)=u
    for r = 2:deg
        % (r) P_r(u) = ((2r-1)u P_{r-1}(u) - (r-1) P_{r-2}(u)) / r
        P{r+1} = ((2*r-1).*u.*P{r} - (r-1).*P{r-1})/r;
    end
end
