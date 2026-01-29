% Function to find the nearest node based on Euclidean distance
function nearest_node = find_nearest_node(longitude, latitude, df_nodes)
    distances = sqrt((df_nodes.x - longitude).^2 + (df_nodes.y - latitude).^2);
    [~, nearest_node] = min(distances);
end