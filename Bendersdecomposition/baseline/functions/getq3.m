function w = getq(distances,lambda,range_threshold,alpha_hat)
for i=1:length(distances)
    for j=1:length(distances)
        d=distances(i,j);
        if(d<range_threshold)
            %w(i,j) = 2/(1+exp(lambda * d*0.6));% function 3     nyc
            w(i,j) = 2/(1+exp(lambda * d*0.8));% function 3
            %w(i,j) = 1.2/(1+0.2*exp(lambda * d*0.6));% function 3 london
        else
            w(i,j) = min(2/(1+exp(lambda * range_threshold*40)),10e-5);% function 3
        end
    end
end
w = alpha_hat*w;
end

