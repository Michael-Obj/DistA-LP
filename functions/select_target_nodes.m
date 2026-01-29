% Function to find nodes within the target region
function [col_osmid_selected, original_longitude, original_latitude, col_longitude_selected, col_latitude_selected] = select_target_nodes(col_longitude, col_latitude, col_osmid, TARGET_LON_MAX, TARGET_LON_MIN, TARGET_LAT_MAX, TARGET_LAT_MIN)
    idx = (col_longitude >= TARGET_LON_MIN) & (col_longitude <= TARGET_LON_MAX) & ...
          (col_latitude >= TARGET_LAT_MIN) & (col_latitude <= TARGET_LAT_MAX);

    fprintf('Number of locations in the selected region: %d\n', length(idx));
    
    % Extract corresponding values
    [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx);

    % Randomly select 10% of the selected locations using rng(0) / Randomly select 20 values using rng(0)
    num_to_select = max(1, floor(0.02 * length(col_osmid_selected))); 
    [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, num_to_select);
end