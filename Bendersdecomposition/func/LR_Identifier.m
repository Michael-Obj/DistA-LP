function user = LR_Identifier(user, env_parameters)
%% The objective of this function is identify the local relevant location of the user

    [~, distance] = shortestpathtree(env_parameters.G, user.loc_ID);        % Calculate the path distance from the user's current location to all the other locations in the road network
    relevant_loc_ID_range = find(distance<env_parameters.GAMMA);            % Find the set of relevant locations within the threshold GAMMA

    %%  Visualize the range of local relevant locations
    % plot(env_parameters.longitude, env_parameters.latitude, '*');
    % hold on; 
    % plot(env_parameters.longitude(relevant_loc_ID_range), env_parameters.latitude(relevant_loc_ID_range), 'o'); 

    LR_loc_ID = randsample(size(relevant_loc_ID_range, 2), user.LR_loc_size); % Sample the set of relevant locations
    user.LR_loc_ID = relevant_loc_ID_range(LR_loc_ID); 
    

    %% Create distance matrix, identify the GeoI neighbors and craete the corresponding GeoI graph
    distance_matrix = zeros(NR_LOC, NR_LOC);                                % Distance between each pair of coordinates 
    GeoI_neighbor = ones(NR_LOC, NR_LOC)*999;                                  % Record the GeoI neighbors
    for i = 1:1:NR_LOC
        for j = 1:1:NR_LOC
            [distance_matrix(i,j), ~, ~] = haversine([longitude(1, i), latitude(1, i)], [longitude(1, j), latitude(1, j)]); % Calculate the Haversine distance between locations
            if distance_matrix(i,j) <= R
                GeoI_neighbor(i, j) = distance_matrix(i,j);
            end
        end
    end
    G_GeoI = digraph(GeoI_neighbor);                                        % Create the GeoI graph using the GeoI_neighbor
end