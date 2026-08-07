function node_in_target = rand_sample(col_longitude,col_latitude,env_parameters)
lon_min = min(col_longitude);
lon_max = max(col_longitude);
lat_min = min(col_latitude);
lat_max = max(col_latitude);

% 2. Calculate the subregion size (one-fifth of the full region here)
lon_range = lon_max - lon_min;
lat_range = lat_max - lat_min;

sub_lon_length = lon_range / 1;
sub_lat_length = lat_range / 1;

% Prepare an empty array for storing points that meet the criteria
valid_idx = [];

% 3. Keep sampling until a subregion containing enough points is found
while length(valid_idx) < env_parameters.NR_NODE_IN_TARGET

    % 3.1 Randomly select the lower-left corner of the subregion
    %     rand() generates a random number in the interval [0,1]
    sub_lon_min = lon_min + rand() * (lon_range - sub_lon_length);
    sub_lon_max = sub_lon_min + sub_lon_length;
    sub_lat_min = lat_min + rand() * (lat_range - sub_lat_length);
    sub_lat_max = sub_lat_min + sub_lat_length;

    % 3.2 Find the indices of all nodes within the subregion
    valid_idx = find( ...
        col_longitude >= sub_lon_min & col_longitude <= sub_lon_max & ...
        col_latitude  >= sub_lat_min & col_latitude  <= sub_lat_max ...
    );

    % If the subregion has too few points, the while loop samples another one
    if length(valid_idx) < env_parameters.NR_NODE_IN_TARGET
        disp('Insufficient points in the randomly selected subregion. Selecting a new subregion...');
    end
end

% 4. Randomly sample the specified number of nodes from the subregion
node_in_target = (valid_idx( randperm(length(valid_idx), env_parameters.NR_NODE_IN_TARGET) ))';
end

