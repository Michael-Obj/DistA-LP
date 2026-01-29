function [G1_all, G2_all, G3_all] = rebuild_from_best_params( ...
                        original_distance_matrices, ...
                        obfuscated_distance_matrices, ...
                        cost_coefficient_matrices, ...
                        best_params)
    % rebuild_from_best_params
    %   Recreate Gaussian matrices from the fitted parameters only
    %   (no permutation vectors).
    %
    % INPUTS  (all cell arrays of equal length)
    %   original_distance_matrices   : {num_locations × 1}  — A1 targets
    %   obfuscated_distance_matrices : {num_locations × 1}  — A2 targets
    %   cost_coefficient_matrices    : {num_locations × 1}  — A3 targets
    %   best_params                  : {num_locations × 1}  — 12-element θ
    %
    % OUTPUTS (cell arrays, same length)
    %   G1_all, G2_all, G3_all : denormalised Gaussian kernels
    
    num_locations = numel(original_distance_matrices);
    
    G1_all = cell(num_locations,1);
    G2_all = cell(num_locations,1);
    G3_all = cell(num_locations,1);
    
    for i = 1:num_locations
    
        % --- fetch target matrices and their sizes -------------------------
        A1 = original_distance_matrices{i};
        A2 = obfuscated_distance_matrices{i};
        A3 = cost_coefficient_matrices{i};
    
        n  = size(A1,1);
        m  = size(A2,2);
    
        % --- build raw Gaussians on a unit grid ----------------------------
        theta = best_params{i};                 % 1×12 vector
        [G1u,G2u,G3u] = generate_gaussians(n, m, theta);
    
        % --- rescale each kernel to match its target matrix ---------------
        G1_all{i} = G1u * (max(A1(:))-min(A1(:))) + min(A1(:));
        G2_all{i} = G2u * (max(A2(:))-min(A2(:))) + min(A2(:));
        G3_all{i} = G3u * (max(A3(:))-min(A3(:))) + min(A3(:));
    
        % --- quick visual sanity-check for the first location -------------
        if i == 1
            figure('Name','Sanity-check for location #1');
            subplot(1,2,1), imagesc(A1), axis image tight, colorbar
            title('A1 original');
    
            subplot(1,2,2), imagesc(G1_all{1}), axis image tight, colorbar
            title('G1 rebuilt from best\_params');
        end
    end
end