function [Hnm,xi,xi_max,xi_change] = getHnm(distance_matrix,epsilon_nmw,EPSILON,user,delta,w)
Hnm=cell(length(distance_matrix), length(distance_matrix));
xi=zeros(length(distance_matrix), length(distance_matrix));
B_xn_xnhat=get_B(w,distance_matrix);
xi_change=[0];
num_nochange=0;
% for n=1:length(distance_matrix)
%     for m=1:length(distance_matrix)
%         if n~=m
%             K=zeros(length(distance_matrix),length(distance_matrix));
%             for i=1:length(distance_matrix)
%                 for j=1:length(distance_matrix)
%                     K(i,j)=(EPSILON-epsilon_nmw(i,j))*distance_matrix(i,j)-(EPSILON-epsilon_nmw(n,m))*distance_matrix(n,m);
%                 end
%             end
%             ldx_left=1;
%             ldx_right=length(K)*length(K);
%             xi(n,m) = binary_search_xi(K, delta, ldx_left, ldx_right,w,n,m,distance_matrix,epsilon_nmw,EPSILON,B_xn_xnhat);
%         end
%     end
% end

%% randomly
num_elements = length(distance_matrix);
[n_values, m_values] = ndgrid(1:num_elements, 1:num_elements);
valid_pairs = [n_values(:), m_values(:)];
valid_pairs = valid_pairs(valid_pairs(:,1) ~= valid_pairs(:,2), :);
random_order = randperm(size(valid_pairs, 1));
random_pairs = valid_pairs(random_order, :);
xi = zeros(num_elements, num_elements); 
for pair_idx = 1:size(random_pairs, 1)
    n = random_pairs(pair_idx, 1);
    m = random_pairs(pair_idx, 2);
    K = zeros(num_elements, num_elements);
    for i = 1:num_elements
        for j = 1:num_elements
            K(i, j) = (EPSILON - epsilon_nmw(i, j)) * distance_matrix(i, j) - (EPSILON - epsilon_nmw(n, m)) * distance_matrix(n, m);
        end
    end
    ldx_left = 1;
    ldx_right = num_elements * num_elements;
    xi(n, m) = binary_search_xi(K, delta, ldx_left, ldx_right, w, n, m, distance_matrix, epsilon_nmw, EPSILON, B_xn_xnhat,Pr);
    if xi_change(end)<xi(n,m)
        num_nochange=0;
        xi_change=[xi_change,xi(n,m)];
    else
        num_nochange=num_nochange+1;
    end
    if num_nochange>1000
        break;
    end
end




xi_max=max(max(xi));
for n=1:length(user)
    for m=1:length(user)
        Hnm{user(n),user(m)}=zeros(length(distance_matrix),length(distance_matrix));
        for i=1:length(distance_matrix)
            for j=1:length(distance_matrix)
                if (EPSILON-epsilon_nmw(i,j))*distance_matrix(i,j)-xi_max<=(EPSILON-epsilon_nmw(user(n),user(m)))*distance_matrix(user(n),user(m))
                    Hnm{user(n),user(m)}(i,j)=1;
                end
            end
        end
    end
end
end