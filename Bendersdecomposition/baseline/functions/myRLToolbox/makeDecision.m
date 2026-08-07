function extreme_idx = makeDecision(policy, result_instant_)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    extreme_array = zeros(1, size(policy.probability, 2)); 
    extreme_array(result_instant_) = 1;
    probability_array = policy.probability.*extreme_array; 
    probability_array_ = cumsum(probability_array); 
    probability_array_ = probability_array_/max(probability_array_); 
    probability_array_ = [0, probability_array_]; 
    seed = rand; 
    for i = 1:1:size(policy.probability, 2)
        if seed > probability_array_(1, i) && seed <= probability_array_(1, i+1)
            extreme_idx = i; 
        end
    end
end