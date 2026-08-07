% function distance_matrix = distanceMatrix(y, x)
%     NR_LOC = size(x, 1); 
%     distance_matrix = zeros(NR_LOC, NR_LOC); 
%     for i = 1:1:NR_LOC
% 
%         for j = i+1:1:NR_LOC
%             if x
%             [distance, ~, ~] = haversine([x(i,1), y(i,1)], [x(j,1), y(j,1)]); 
%             % if distance <= NEIGHBOR_THRESHOLD
%                 distance_matrix(i, j) = distance;
%                 distance_matrix(j, i) = distance;
%             % end
%         end
%     end
% end


function distance_matrix = distanceMatrix(longitude, latitude)

    lon = longitude(:);
    lat = latitude(:);
    

    lat_rad = lat * pi / 180;
    lon_rad = lon * pi / 180;
    

    n = length(lat);
    

    distance_matrix = zeros(n, n);
    

    for i = 1:n
  
        delta_lat = lat_rad - lat_rad(i);
        delta_lon = lon_rad - lon_rad(i);
        
 
        a = sin(delta_lat/2).^2 + cos(lat_rad(i)) * cos(lat_rad) .* sin(delta_lon/2).^2;
        c = 2 * atan2(sqrt(a), sqrt(1-a));
        
   
        R = 6371;
        distance_matrix(i, :) = R * c;
    end
end