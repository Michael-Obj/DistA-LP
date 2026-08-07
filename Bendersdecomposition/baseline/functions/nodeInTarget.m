function node_in_target = nodeInTarget(x, y, TARGET_X_MAX, TARGET_X_MIN, TARGET_Y_MAX, TARGET_Y_MIN)
    NR_LOC = size(x, 1); 
    node_in_target = [];
    for i = 1:1:NR_LOC
        if x(i, 1) >= TARGET_X_MIN && x(i, 1) <= TARGET_X_MAX
            if y(i, 1) >= TARGET_Y_MIN && y(i, 1) <= TARGET_Y_MAX
                node_in_target = [node_in_target i]; 
            end
        end
    end
end