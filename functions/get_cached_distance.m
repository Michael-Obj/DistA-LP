function km = get_cached_distance(loc1, loc2)
    % Memoized distance computation using a global cache
    % loc1 and loc2 are 1x2 vectors: [lat, lon]

    % Ensure global cache is available
    global distance_cache;
    if isempty(distance_cache)
        distance_cache = containers.Map();
    end

    % Create symmetric key
    key1 = sprintf('%.6f,%.6f,%.6f,%.6f', loc1(1), loc1(2), loc2(1), loc2(2));
    key2 = sprintf('%.6f,%.6f,%.6f,%.6f', loc2(1), loc2(2), loc1(1), loc1(2));

    if isKey(distance_cache, key1)
        km = distance_cache(key1);
    elseif isKey(distance_cache, key2)
        km = distance_cache(key2);
    else
        km = haversine(loc1, loc2);  % Use your existing haversine.m
        distance_cache(key1) = km;
    end
end







% function d = get_cached_distance(lat1, lon1, lat2, lon2)
%     global distance_cache;
% 
%     % Generate a symmetric key (order of points doesn't matter)
%     key = sprintf('%.6f,%.6f,%.6f,%.6f', lat1, lon1, lat2, lon2);
%     reverse_key = sprintf('%.6f,%.6f,%.6f,%.6f', lat2, lon2, lat1, lon1);
% 
%     if isKey(distance_cache, key)
%         d = distance_cache(key);
%     elseif isKey(distance_cache, reverse_key)
%         d = distance_cache(reverse_key);
%     else
%         d = haversine_distance(lat1, lon1, lat2, lon2);
%         distance_cache(key) = d;
%     end
% end
