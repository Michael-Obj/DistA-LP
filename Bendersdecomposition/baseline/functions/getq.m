function w = getq(distances,lambda,range_threshold,alpha_hat)
for i=1:length(distances)
    for j=1:length(distances)
        d=distances(i,j);
        if(d<range_threshold)
            %w(i,j) = exp(-lambda * d*0.4); % function 1   nyc    range_threshold = D_MAX/30;
            w(i,j) = exp(-lambda * d*0.4); % function 1
        else
            %w(i,j) = exp(-lambda * range_threshold*40);%w(i,j) = exp(-lambda * range_threshold*20);
            w(i,j) = min(exp(-lambda * range_threshold*40),10e-5);%5000
        end
    end
end
w = alpha_hat*w;
end

