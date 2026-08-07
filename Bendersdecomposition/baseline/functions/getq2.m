function w = getq(distances,lambda,range_threshold,alpha_hat)
for i=1:length(distances)
    for j=1:length(distances)
        d=distances(i,j);
        if(d<range_threshold)       
            %w(i,j) = 0.7/(1+0.3*(d^(lambda*40)));% function 2  nyc        
            w(i,j) = 1/(1+0.75*(d^(lambda)));% function 2  rome  
            %w(i,j) = 1/(1+0.001*(d^(lambda*20)));% function 2  london
        else
            w(i,j) = min(1/(1+exp(10)*(range_threshold+1)^(lambda*10)),10e-5);% function 2
        end
    end
end
w = alpha_hat*w;
end

