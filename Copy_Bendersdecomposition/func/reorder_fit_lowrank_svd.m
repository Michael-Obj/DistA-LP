% ======================================================================
% reorder_fit_lowrank_svd.m
% ----------------------------------------------------------------------
%  Fit **low‑rank (truncated SVD)** surrogate surfaces for the three target
%  matrices used in your distance–privacy pipeline, while simultaneously
%  searching for an informative row permutation.  Signature matches your
%  existing helpers (polynomial / Gaussian / RBF) so you can plug‑and‑play.
%
%     [best_pi , best_factors , F1 , F2 , F3] = reorder_fit_lowrank_svd( ...
%              A1 , A2 , A3 , rank_r , lambda2 , lambda3 )
%
%  INPUTS
%  -------
%    • A1 : n×n   symmetric matrix  (original–vs–original distances)
%    • A2 : n×m   matrix            (original–vs–obfuscated distances)
%    • A3 : n×m   matrix            (cost‑coefficients)
%    • rank_r   : target rank of the truncated SVD   (default = 5)
%    • lambda2  : weight for A2 term in L¹ loss       (default = 1)
%    • lambda3  : weight for A3 term in L¹ loss       (default = 1)
%
%  OUTPUTS
%  --------
%    • best_pi      : 1×n permutation of the rows/cols giving minimal loss
%    • best_factors : struct with compact SVD factors for the three surfaces
%    • F1 , F2 , F3 : reconstructed surfaces (n×n , n×m , n×m) **after the
%                     permutation is undone**, so they align with A1/A2/A3
%
%  NOTE —  Noise injection is **not** included here; re‑use your existing
%  `add_noise_to_distance_matrix` (or singular‑value perturbation) exactly
%  as you do for the other models.
% ======================================================================

function [best_pi , best_factors , F1 , F2 , F3] = reorder_fit_lowrank_svd( ...
            A1 , A2 , A3 , rank_r , lambda2 , lambda3 )

    % -------------------- defaults & sanity checks --------------------
    if nargin < 4 || isempty(rank_r),   rank_r  = 5;   end
    if nargin < 5 || isempty(lambda2),  lambda2 = 1;   end
    if nargin < 6 || isempty(lambda3),  lambda3 = 1;   end

    [n, m] = size(A2);                               % n rows, m cols
    rank_r = min(rank_r, min([n, m]));               % cannot exceed dims

    % -------------------- helper: truncated SVD -----------------------
    function [U,S,V] = trunc_svd(M, r)
        % economy‑size SVD then keep first r comps
        [U,S,V] = svd(M, 'econ');
        U = U(:,1:r);
        S = S(1:r,1:r);
        V = V(:,1:r);
    end

    % -------------------- helper: build surfaces ----------------------
    function [fac , F1p , F2p , F3p] = build_surfaces(M1, M2, M3)
        [U1,S1,V1] = trunc_svd(M1, rank_r);
        [U2,S2,V2] = trunc_svd(M2, rank_r);
        [U3,S3,V3] = trunc_svd(M3, rank_r);

        F1p = U1 * S1 * V1';        % n × n
        F2p = U2 * S2 * V2';        % n × m
        F3p = U3 * S3 * V3';        % n × m

        fac = struct('U1',U1,'S1',S1,'V1',V1, ...
                      'U2',U2,'S2',S2,'V2',V2, ...
                      'U3',U3,'S3',S3,'V3',V3);
    end

    % -------------------- helper: L¹ loss -----------------------------
    function L = loss(M1, M2, M3, surf1, surf2, surf3)
        L = sum(abs(M1(:) - surf1(:))) + ...
            lambda2 * sum(abs(M2(:) - surf2(:))) + ...
            lambda3 * sum(abs(M3(:) - surf3(:)));
    end

    % -------------------- initial permutation = identity -------------
    pi_curr = 1:n;
    [fac_curr , F1p , F2p , F3p] = build_surfaces(A1, A2, A3);   % no perm yet
    loss_curr = loss(A1, A2, A3, F1p, F2p, F3p);

    % -------------------- greedy swap + re‑fit loop -------------------
    MAX_ITER = 10;
    for iter = 1:MAX_ITER
        improved = false;
        for i = 1:n
            for j = i+1:n
                % try swapping rows i & j (and cols in A1)
                pi_try = pi_curr;
                pi_try([i j]) = pi_try([j i]);

                A1p = A1(pi_try, pi_try);
                A2p = A2(pi_try, :);
                A3p = A3(pi_try, :);

                [fac_try , F1t , F2t , F3t] = build_surfaces(A1p, A2p, A3p);
                loss_try = loss(A1p, A2p, A3p, F1t, F2t, F3t);

                if loss_try < loss_curr
                    % accept swap + new factors
                    pi_curr   = pi_try;
                    fac_curr  = fac_try;
                    F1p       = F1t;   F2p = F2t;   F3p = F3t;
                    loss_curr = loss_try;
                    improved  = true;
                end
            end
        end
        if ~improved,  break;  end
    end

    best_pi      = pi_curr;
    best_factors = fac_curr;

    % -------------------- undo permutation so outputs align ----------
    inv_pi = zeros(1,n);   inv_pi(best_pi) = 1:n;
    F1 = F1p(inv_pi, inv_pi);  % symmetric n×n
    F2 = F2p(inv_pi, :);       % n×m
    F3 = F3p(inv_pi, :);       % n×m

    % -------------------- verbose summary ----------------------------
    fprintf('[LowRank‑SVD]  Best permutation  :  [%s]\n', num2str(best_pi));
    fprintf('[LowRank‑SVD]  Rank r            :  %d\n', rank_r);
    fprintf('[LowRank‑SVD]  Minimum L1 loss   :  %.4f\n', loss_curr);
end
% ======================================================================
