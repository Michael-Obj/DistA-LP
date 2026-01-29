% function [best_pi, best_params, F1, F2, F3] = reorder_fit_gaussians(A1, A2, A3, lambda2, lambda3)
%     % Input:
%     %   A1: n x n symmetric matrix
%     %   A2, A3: n x m matrices
%     %   lambda2, lambda3: weighting terms
%     % Output:
%     %   best_pi: optimal permutation indices
%     %   best_params: Gaussian parameters
%     %   F1, F2, F3: learned Gaussian target functions
% 
% 
%     [n, m] = size(A2);
%     pi_current = 1:n; % Start with identity permutation
% 
% 
%     % Random initial Gaussian parameters
%     params_current = [
%         n/2, n/2, n/4, n/4, ... % F1
%         n/2, m/2, n/4, m/4, ... % F2
%         n/2, m/2, n/4, m/4  % F3
%     ];
% 
% 
%     % Optimize Gaussian parameters first
%     options = optimset('Display', 'off', 'MaxFunEvals', 1000);
%     params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), params_current, options);
% 
% 
%     loss_current = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);
% 
% 
%     max_iter = 10;
%     for iter = 1:max_iter
%         improved = false;
% 
% 
%         % Try swapping two indices to improve permutation
%         for i = 1:n
%             for j = i+1:n
%                 pi_trial = pi_current;
%                 pi_trial([i j]) = pi_trial([j i]); % Swap i and j
% 
% 
%                 loss_trial = compute_loss(A1, A2, A3, pi_trial, params_current, lambda2, lambda3);
% 
% 
%                 if loss_trial < loss_current
%                     pi_current = pi_trial;
%                     loss_current = loss_trial;
%                     improved = true;
%                 end
%             end
%         end
% 
% 
%         % Re-optimize Gaussian parameters given new permutation
%         params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), params_current, options);
%         loss_current = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);
% 
% 
%         if ~improved
%             break; % No permutation improvement, done
%         end
%     end
% 
% 
%     best_pi = pi_current;
%     best_params = params_current;
% 
% 
%     % Generate final learned Gaussians
%     [F1, F2, F3] = generate_gaussians(n, m, best_params);
% 
% 
%     fprintf('Best permutation: [%s]\n', num2str(best_pi));
%     fprintf('Best Gaussian params:\n');
%     disp(reshape(best_params, 4, 3)');
%     fprintf('Minimum L1 loss: %.4f\n', loss_current);
% 
% 
%     % Visualization
%     A1_reordered = A1(best_pi, best_pi);
%     A2_reordered = A2(best_pi, :);
%     A3_reordered = A3(best_pi, :);
% 
% 
%     % figure('Name', 'Matrix Reordering (Smart Optimization)', 'Position', [100, 100, 1600, 800]);
%     % 
%     % 
%     % subplot(3, 4, 1); imagesc(A1); colorbar; title('Original A_1');
%     % subplot(3, 4, 2); imagesc(F1); colorbar; title('Learned Target F_1');
%     % subplot(3, 4, 3); imagesc(A1_reordered); colorbar; title('Permuted A_1');
%     % 
%     % 
%     % subplot(3, 4, 5); imagesc(A2); colorbar; title('Original A_2');
%     % subplot(3, 4, 6); imagesc(F2); colorbar; title('Learned Target F_2');
%     % subplot(3, 4, 7); imagesc(A2_reordered); colorbar; title('Permuted A_2');
%     % 
%     % 
%     % subplot(3, 4, 9); imagesc(A3); colorbar; title('Original A_3');
%     % subplot(3, 4, 10); imagesc(F3); colorbar; title('Learned Target F_3');
%     % subplot(3, 4, 11); imagesc(A3_reordered); colorbar; title('Permuted A_3');
% end
% 
% 
% function loss = compute_loss(A1, A2, A3, pi, params, lambda2, lambda3)
%     [n, m] = size(A2);
% 
% 
%     [F1, F2, F3] = generate_gaussians(n, m, params);
% 
% 
%     A1p = A1(pi, pi);
%     A2p = A2(pi, :);
%     A3p = A3(pi, :);
% 
% 
%     loss1 = sum(abs(A1p(:) - F1(:)));
%     loss2 = sum(abs(A2p(:) - F2(:)));
%     loss3 = sum(abs(A3p(:) - F3(:)));
% 
% 
%     loss = loss1 + lambda2 * loss2 + lambda3 * loss3;
% end
% 
% 
% function [F1, F2, F3] = generate_gaussians(n, m, params)
%     [I1, J1] = meshgrid(1:n, 1:n);
%     [I2, J2] = meshgrid(1:n, 1:m);
% 
% 
%     F1 = gaussian2d(I1, J1, params(1), params(2), params(3), params(4));
%     F2 = gaussian2d(I2, J2, params(5), params(6), params(7), params(8));
%     F3 = gaussian2d(I2, J2, params(9), params(10), params(11), params(12));
% end
% 
% 
% function val = gaussian2d(x, y, cx, cy, sx, sy)
%     val = exp(-((x - cx).^2 ./ (2 * sx^2) + (y - cy).^2 ./ (2 * sy^2)));
% end






function [best_pi, best_params, F1, F2, F3] = reorder_fit_gaussians(A1, A2, A3, lambda2, lambda3)
    % Input:
    %   A1: n x n symmetric matrix
    %   A2, A3: n x m matrices
    %   lambda2, lambda3: weighting terms
    % Output:
    %   best_pi: optimal permutation indices
    %   best_params: Gaussian parameters (6 per component)
    %   F1, F2, F3: learned Gaussian target functions

    [n, m] = size(A2);
    pi_current = 1:n; % Start with identity permutation

    % Random initial Gaussian parameters [cx, cy, sx, sy, A, b] for each of 3 components
    params_current = [
        n/2, n/2, n/4, n/4, 1.0, 0.0, ... % F1
        n/2, m/2, n/4, m/4, 1.0, 0.0, ... % F2
        n/2, m/2, n/4, m/4, 1.0, 0.0      % F3
    ];

    % Optimize Gaussian parameters first
    options = optimset('Display', 'off', 'MaxFunEvals', 1000);
    params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), params_current, options);
    loss_current = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);

    max_iter = 10;
    for iter = 1:max_iter
        improved = false;

        % Try swapping two indices to improve permutation
        for i = 1:n
            for j = i+1:n
                pi_trial = pi_current;
                pi_trial([i j]) = pi_trial([j i]); % Swap i and j

                loss_trial = compute_loss(A1, A2, A3, pi_trial, params_current, lambda2, lambda3);
                if loss_trial < loss_current
                    pi_current = pi_trial;
                    loss_current = loss_trial;
                    improved = true;
                end
            end
        end

        % Re-optimize Gaussian parameters given new permutation
        params_current = fminsearch(@(p) compute_loss(A1, A2, A3, pi_current, p, lambda2, lambda3), params_current, options);
        loss_current = compute_loss(A1, A2, A3, pi_current, params_current, lambda2, lambda3);

        if ~improved
            break; % No permutation improvement, done
        end
    end

    best_pi = pi_current;
    best_params = params_current;

    % Generate final learned Gaussians
    [F1, F2, F3] = generate_gaussians(n, m, best_params);

    fprintf('Best permutation: [%s]\n', num2str(best_pi));
    fprintf('Best Gaussian params (6 per component):\n');
    disp(reshape(best_params, 6, 3)');
    fprintf('Minimum L1 loss: %.4f\n', loss_current);
end

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

    % Unpack parameters for each Gaussian
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
