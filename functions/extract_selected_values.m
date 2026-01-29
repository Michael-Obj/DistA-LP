% Function to extract corresponding values
function [col_osmid_selected, col_longitude_selected, col_latitude_selected] = extract_selected_values(col_osmid, col_longitude, col_latitude, idx)
    col_osmid_selected = col_osmid(idx);
    col_longitude_selected = col_longitude(idx);
    col_latitude_selected = col_latitude(idx);
end