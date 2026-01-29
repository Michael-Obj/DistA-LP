% Function to randomly select N nodes using rng(0)
function [col_osmid_selected, original_longitude, original_latitude] = select_random_n(col_osmid_selected, col_longitude_selected, col_latitude_selected, n)
    rng(0);  % Set the random seed for reproducibility
    num_selected = min(n, length(col_osmid_selected));  % Ensure we don't select more than available data

    % Randomly select indices
    random_indices = randperm(length(col_osmid_selected), num_selected);
    
    % Select the random nodes
    col_osmid_selected = col_osmid_selected(random_indices);
    original_longitude = col_longitude_selected(random_indices);
    original_latitude = col_latitude_selected(random_indices);
end