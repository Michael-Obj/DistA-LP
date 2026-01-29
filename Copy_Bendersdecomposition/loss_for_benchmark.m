function [loss_benchmarks,coarse_loss,P_matrix]=loss_for_benchmark(env_parameters, obf_loc, distance_matrix_original, node_in_target, G, task_loc)


NR_LOC = length(distance_matrix_original); 
NR_OBFLOC = size(obf_loc, 2); 
cost_matrix = zeros(NR_LOC, NR_OBFLOC); 
for i = 1:1:NR_LOC
    [~, D] = shortestpathtree(G, node_in_target(task_loc)); % road
    %[~, D] = shortestpathtree(G, 1); % grid
    for j = 1:1:NR_OBFLOC         
        cost_matrix(i,j) = abs(D(node_in_target(i))-D(node_in_target(obf_loc(j)))); 
    end
end
cost = cost_matrix/NR_LOC; 




P_matrix=zeros(length(distance_matrix_original),length(obf_loc));
sum_i=zeros(length(distance_matrix_original),1);
epsilon=env_parameters.EPSILON;
for i=1:length(distance_matrix_original)
    for j=1:length(obf_loc)
        sum_i(i,1)=sum_i(i,1)+exp(-epsilon*distance_matrix_original(i,obf_loc(j))/2.0);
    end
    for j=1:length(obf_loc)
        P_matrix(i,j)=exp(-epsilon*distance_matrix_original(i,obf_loc(j))/2.0)/sum_i(i,1);
    end
end
loss_benchmarks = sum(sum(cost .* P_matrix));%/length(node_in_target);

tic;
P_2=zeros(length(obf_loc),length(distance_matrix_original));
for i=1:length(obf_loc)
    for j=1:length(distance_matrix_original)
        P_2(i,j)=P_matrix(j,i)/sum(P_matrix(:,i));
    end
end
y_k=sparse(length(distance_matrix_original),0);
for i=1:length(obf_loc)
    sum_pc=[];
    for j=1:length(obf_loc)
        sum_pc_j=P_2(i,:)*cost(:,j);
        sum_pc=[sum_pc,sum_pc_j];
    end
    [min_sum, y_k(i)] = min(sum_pc);
end

coarse_loss=sum(sum(cost(:,y_k) .* P_matrix));
time_coarse=toc;
end




