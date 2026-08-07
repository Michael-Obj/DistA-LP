function [xi_hathat,xi_real] = get_xi_hathat(distance_matrix,epsilon_nmw,EPSILON,user,delta,w,Pr, all_target,threshold_matrix,B_xn_xnhat,range_threshold)
Hnm=cell(length(distance_matrix), length(distance_matrix));
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

%% pre calculate K

num_elements = length(all_target);
xi_hathat = zeros(num_elements, num_elements); 
xi_real = zeros(num_elements, num_elements);

K=cell(length(distance_matrix),length(distance_matrix));
K_base=sparse(length(distance_matrix),length(distance_matrix));
% for i = 1:length(distance_matrix)
%     for j = 1:length(distance_matrix)
%         K_base(i, j) = (EPSILON - epsilon_nmw(i,j)) * distance_matrix(i,j);
%     end
% end
K_base = (max(0,EPSILON - epsilon_nmw)) .* distance_matrix;

% for n=1:length(distance_matrix)
%     for m=1:length(distance_matrix)
%         K{n,m}=sparse(length(distance_matrix),length(distance_matrix));
%         row_indices = find(threshold_matrix(n,:) == 1); 
%         col_indices = find(threshold_matrix(m,:) == 1);
%         K{n,m}(row_indices, col_indices) = K_base(row_indices, col_indices) - (EPSILON - epsilon_nmw(n,m)) * distance_matrix(n,m);
%     end
% end

% N = size(distance_matrix, 1);
% indices = cell(N, 1);
% for i = 1:N
%     indices{i} = find(threshold_matrix(i, :) == 1);
% end
% K = cell(N, N);
% for n = 1:N
%     row_idx = indices{n};
%     for m = 1:N
%         col_idx = indices{m};
%         scalarVal = (EPSILON - epsilon_nmw(n, m)) * distance_matrix(n, m);
%         subK = K_base(row_idx, col_idx) - scalarVal;
%         [RR, CC] = ndgrid(row_idx, col_idx);
%         K{n, m} = sparse(RR(:), CC(:), subK(:), N, N);
%     end
% end

N = size(distance_matrix, 1);
indices = cell(N, 1);
for i = 1:N
    indices{i} = find(threshold_matrix(i, :) == 1);
end
% K = cell(N, N);
% for n = 1:N
%     row_idx = indices{n};
%     for m = 1:N
%         col_idx = indices{m};
%         scalarVal = (EPSILON - epsilon_nmw(n, m)) * distance_matrix(n, m);
%         nnz_est = numel(row_idx) * numel(col_idx);   
%         K_nm = spalloc(N, N, nnz_est);
%         K_nm(row_idx, col_idx) = K_base(row_idx, col_idx) - scalarVal;
%         K{n, m} = K_nm;
%     end
% end
% 
% row_idx = indices{n_hat};
% col_idx = indices{m_hat};
% scalarVal = (EPSILON - epsilon_nmw(n_hat, m_hat)) * distance_matrix(n_hat, m_hat);
% nnz_est = numel(row_idx) * numel(col_idx);   
% K_nm = spalloc(N, N, nnz_est);
% K_nm(row_idx, col_idx) = K_base(row_idx, col_idx) - scalarVal;
% k_nhat_mhat=K_nm;


%% pre calculate xi table
%xi_table=K;


%%

for n_hat_ind=1:length(all_target)
    n_hat=all_target(n_hat_ind);
    
    for m_hat_ind=1:length(all_target)
        m_hat=all_target(m_hat_ind);
        [n_hat_ind, m_hat_ind]
        
        if n_hat~=m_hat

            % num_nearest_n=length(find(w(n_hat,:)>0.01));
            % [~, sortedIdx] = sort(distance_matrix(n_hat,:), 'ascend');
            % closest_nhat = sortedIdx(1:num_nearest_n);
            % num_nearest_m=length(find(w(m_hat,:)>0.01));

            num_nearest_n=length(find(distance_matrix(n_hat,:)<0.5*range_threshold));
            %num_nearest_n=1;
            [~, sortedIdx] = sort(distance_matrix(n_hat,:), 'ascend');
            closest_nhat = sortedIdx(1:num_nearest_n);  

            num_nearest_m=length(find(distance_matrix(m_hat,:)<0.5*range_threshold));
            %num_nearest_m=1;
            [~, sortedIdx] = sort(distance_matrix(m_hat,:), 'ascend');
            closest_mhat = sortedIdx(1:num_nearest_m);

            ldx_left = 1;

            row_idx = indices{n_hat};
            col_idx = indices{m_hat};
            scalarVal = (max(0,EPSILON - epsilon_nmw(n_hat, m_hat))) * distance_matrix(n_hat, m_hat);
            nnz_est = numel(row_idx) * numel(col_idx);   
            K_nm = spalloc(N, N, nnz_est);
            K_nm(row_idx, col_idx) = K_base(row_idx, col_idx) - scalarVal;
            k_nhat_mhat=K_nm;

            result = k_nhat_mhat;
            xi_table_nhat_m_hat=k_nhat_mhat;
            [row_idx, col_idx, values] = find(result);
            [sorted_values, sort_idx] = sort(values);
            row_idx_k_sort = row_idx(sort_idx);
            col_idx_k_sort = col_idx(sort_idx);
            ldx_right=length(sorted_values);
            xi_real(n_hat_ind,m_hat_ind)=linear_search_xi(sorted_values, ldx_left, ldx_right, B_xn_xnhat(n_hat,:),B_xn_xnhat(m_hat,:),(1 - delta)/Pr(n_hat,m_hat),xi_table_nhat_m_hat,row_idx_k_sort, col_idx_k_sort);
            for i = 1:num_nearest_n
                for j = 1:num_nearest_m
                    % K(i, j) = (EPSILON - epsilon_nmw(closest_nhat(i), closest_mhat(j))) * distance_matrix(closest_nhat(i), closest_mhat(j)) - (EPSILON - epsilon_nmw(n_hat, m_hat)) * distance_matrix(n_hat, m_hat);
                    % new_xi=binary_search_xi(K{closest_nhat(i), closest_mhat(j)}, delta, ldx_left, ldx_right, w, n_hat, m_hat, distance_matrix, epsilon_nmw, EPSILON, B_xn_xnhat(n_hat,:),B_xn_xnhat(m_hat,:),Pr,xi_table{n_hat, m_hat});
                    
                    row_idx = indices{closest_nhat(i)};
                    col_idx = indices{closest_mhat(j)};
                    scalarVal = (max(0,EPSILON - epsilon_nmw(closest_nhat(i), closest_mhat(j)))) * distance_matrix(closest_nhat(i), closest_mhat(j));
                    nnz_est = numel(row_idx) * numel(col_idx);   
                    K_nm = spalloc(N, N, nnz_est);
                    K_nm(row_idx, col_idx) = K_base(row_idx, col_idx) - scalarVal;
                                        
                    result = K_nm;
                    xi_table_nhat_m_hat = K_nm;
                    [row_idx, col_idx, values] = find(result);
                    [sorted_values, sort_idx] = sort(values);
                    row_idx_k_sort = row_idx(sort_idx);
                    col_idx_k_sort = col_idx(sort_idx);
                    ldx_right=length(sorted_values);
                    if isempty(sorted_values)
                        new_xi=0;
                    else
                        Pr(n_hat,m_hat)=1;
                        new_xi=linear_search_xi(sorted_values, ldx_left, ldx_right, B_xn_xnhat(n_hat,:),B_xn_xnhat(m_hat,:),(1 - delta)/Pr(n_hat,m_hat),xi_table_nhat_m_hat,row_idx_k_sort, col_idx_k_sort);
                        %xi_hathat(n_hat_ind,m_hat_ind)=new_xi+xi_hathat(n_hat_ind,m_hat_ind);
                        xi_hathat(n_hat_ind,m_hat_ind)=max(new_xi,xi_hathat(n_hat_ind,m_hat_ind));
                    end
                end
            end
            % xi_hathat(n_hat_ind,m_hat_ind)=xi_hathat(n_hat_ind,m_hat_ind)/(num_nearest_n*num_nearest_m);
        end
        
    end
end


% for pair_idx = 1:size(random_pairs, 1)
%     n = random_pairs(pair_idx, 1);
%     m = random_pairs(pair_idx, 2);
%     K = zeros(num_elements, num_elements);
%     for i = 1:num_elements
%         for j = 1:num_elements
%             K(i, j) = (EPSILON - epsilon_nmw(i, j)) * distance_matrix(i, j) - (EPSILON - epsilon_nmw(n, m)) * distance_matrix(n, m);
%         end
%     end
%     ldx_left = 1;
%     ldx_right = num_elements * num_elements;
%     xi_hathat(n, m) = binary_search_xi(K, delta, ldx_left, ldx_right, w, n, m, distance_matrix, epsilon_nmw, EPSILON, B_xn_xnhat);
% 
% end
end