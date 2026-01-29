% Function to compute raw distance matrix using memoization
function raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude)
    num_original = length(original_longitude);
    num_obfuscated = length(obfuscated_longitude);
    raw_distance_matrix = zeros(num_original, num_obfuscated);

    for i = 1:num_original
        for j = 1:num_obfuscated
            loc1 = [original_latitude(i), original_longitude(i)];
            loc2 = [obfuscated_latitude(j), obfuscated_longitude(j)];
            raw_distance_matrix(i, j) = get_cached_distance(loc1, loc2);  % Uses memoized wrapper
        end
    end
end




% % Function to compute raw distance matrix
% function raw_distance_matrix = compute_raw_distance_matrix(original_longitude, original_latitude, obfuscated_longitude, obfuscated_latitude)
%     num_original = length(original_longitude);
%     num_obfuscated = length(obfuscated_longitude);
%     raw_distance_matrix = zeros(num_original, num_obfuscated);
%     for i = 1:num_original
%         for j = 1:num_obfuscated
%             % raw_distance_matrix(i, j) = haversine_calc(original_latitude(i), original_longitude(i), obfuscated_latitude(j), obfuscated_longitude(j));
%             raw_distance_matrix(i, j) = haversine([original_latitude(i), original_longitude(i)], [obfuscated_latitude(j), obfuscated_longitude(j)]);
% 
%         end
%     end
% end
