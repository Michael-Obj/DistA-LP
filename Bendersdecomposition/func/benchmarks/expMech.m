function [cost, time] = expMech(user, env_parameters)
%% Description:
    % The obfmatrix_generator_laplace is function which generate the
    % obfuscaed location vector
%% Input
    % top_loc_list: Actual x and y coordinate from the openstreet map data
    % i: location on which obfuscation will happen
    % EPSILON: randomized value
    % NR_CANDIDATE: total number of nodes

%% Output

    tic 
    obfuscationMatrix = zeros(user.LR_loc_size, size(user.obf_loc_ID, 2));
    for i = 1:1:user.LR_loc_size
        for j = 1:1:size(user.obf_loc_ID, 2)
            obfuscationMatrix(i, j) = exp(-user.distance_matrix_LR2obf(i,j)*env_parameters.EPSILON/2);        % changed i to 1
        end
        obfuscationMatrix(i, :) = obfuscationMatrix(i, :)/sum(obfuscationMatrix(i, :));            %changed i to 1
    end
    time = toc; 
    cost = 0;
    for i = 1:1:user.LR_loc_size
        for j = 1:1:size(user.obf_loc_ID, 2)
            cost = cost + user.cost_matrix_whole(i,j)*obfuscationMatrix(i, j); 
        end
    end
end
