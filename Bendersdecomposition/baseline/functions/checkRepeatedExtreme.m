function isRepeated = checkRepeatedExtreme(extremerays, u)
    isRepeated = 0;
    for i = 1:1:size(extremerays, 2)
        u_compare = extremerays(:, i);
        difference = sum(abs(u_compare - u));
        if difference <= 0.00001
            isRepeated = 1;
            return; 
        end
    end
end