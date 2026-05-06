function [cost, time] = lapMech(user, env_parameters)
%% Description:
    % The obfmatrix_generator_laplace is function which generate the
    % obfuscaed location vector
%% Input
    % top_loc_list: Actual x and y coordinate from the openstreet map data
    % i: location on which obfuscation will happen
    % EPSILON: randomized value
    % NR_CANDIDATE: total number of nodes

%% Output
    time = 0; 

    obfuscationMatrix = zeros(user.LR_loc_size, size(user.obf_loc_ID, 2));
    
    p = rand(1, 100);
    noise_length = -1/(env_parameters.EPSILON*100)*(lambertw(-1, (p-1)/2.71828)+1);
    noise_angle = rand(1, 100)*2*3.14;

    noise_latitude = noise_length.*cos(noise_angle)/111.320; 
    noise_longitude = noise_length.*sin(noise_angle)/110.574; 


    for i = 1:1:user.LR_loc_size
        for j = 1:1:100
            obfuscatedlocations_long(i, j) = env_parameters.longitude_selected(user.LR_loc_ID(1, i)) + noise_longitude(1, j);
            obfuscatedlocations_lati(i, j) = env_parameters.latitude_selected(user.LR_loc_ID(1, i)) + noise_latitude(1, j);
            for l = 1:1:size(user.obf_loc_ID, 2)
                real_loc = [obfuscatedlocations_long(i, j), obfuscatedlocations_lati(i, j)]; 
                obf_loc = [env_parameters.longitude_selected(user.obf_loc_ID(1, l), 1), env_parameters.latitude_selected(user.obf_loc_ID(1, l), 1)];
                distance(l) = haversine(real_loc, obf_loc); 
            end
            [~, obfuscatedlocations_idx(i, j)] = min(distance); 
        end
    end
    
    cost = 0;
    for i = 1:1:user.LR_loc_size
        for j = 1:1:size(user.obf_loc_ID, 2)
            cost = cost + user.cost_matrix_whole(i, obfuscatedlocations_idx(i, j)); 
        end
    end
    cost = cost/(20*100); 
end
