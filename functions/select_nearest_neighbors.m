% Function to select nearest neighbors for a given location
function [nearest_longitude, nearest_latitude] = select_nearest_neighbors(longitudes, latitudes, idx, num_neighbors)
    distances = zeros(length(longitudes), 1);
    for i = 1:length(longitudes)
        if i ~= idx
            distances(i) = haversine_calc(latitudes(idx), longitudes(idx), latitudes(i), longitudes(i));
        else
            distances(i) = inf; % Exclude the location itself
        end
    end
    
    [~, sorted_indices] = sort(distances);
    nearest_longitude = longitudes(sorted_indices(1:num_neighbors));
    nearest_latitude = latitudes(sorted_indices(1:num_neighbors));
end
