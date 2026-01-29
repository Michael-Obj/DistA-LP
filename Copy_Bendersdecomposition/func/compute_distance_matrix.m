function distance_matrix = compute_distance_matrix(original_latitude, original_longitude)
    num_original = length(original_longitude);
    distance_matrix = zeros(num_original, num_original);
    for i = 1:num_original
        for j = 1:num_original
            distance_matrix(i, j) = haversine_calc(original_latitude(i), original_longitude(i), original_latitude(j), original_longitude(j));
            % distance_matrix(i, j) = haversine([original_latitude(i), original_longitude(i)], [original_latitude(j), original_longitude(j)]);

        end
    end
end
