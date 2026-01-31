function [coordinate, edge, G, NR_LOC] = read_data(target)


%% Building
if strcmp(target, 'building')
   [coordinate, edge, edge_weight] = read_building_data();
end

%% Campus 
if strcmp(target, 'campus')
    [coordinate, edge, edge_weight] = read_campus_data();
end 
    
    G = digraph([edge(:, 1); edge(:, 2)], [edge(:, 2); edge(:, 1)], [edge_weight; edge_weight]);
    NR_LOC = size(coordinate, 1);
end