function [new_adjacence_matrix, new_distance_matrix, new_beta] = reget(adjacence_matrix, distance_matrix, all_target, beta)
%% get new matrix
new_distance_matrix=zeros(length(all_target),length(all_target));
new_adjacence_matrix=zeros(length(all_target),length(all_target));
new_beta=zeros(length(all_target),length(all_target));
for i=1:length(all_target)
    for j=1:length(all_target)
        new_distance_matrix(i,j)=distance_matrix(all_target(i),all_target(j));
        new_adjacence_matrix(i,j)=adjacence_matrix(all_target(i),all_target(j));
        new_beta(i,j)=beta(all_target(i),all_target(j));
    end
end
end

