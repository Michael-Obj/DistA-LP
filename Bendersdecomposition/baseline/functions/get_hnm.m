function hnm=get_hnm(xi_l,w,n,m,distance_matrix,epsilon_nmw,EPSILON,B_xn_xnhat,xi_table)
% hnm=0;
% for n_hat=1:length(distance_matrix)
%     for m_hat=1:length(distance_matrix)
%         if (EPSILON-epsilon_nmw(n_hat,m_hat))*distance_matrix(n_hat,m_hat)-delta_l<=(EPSILON-epsilon_nmw(n,m))*distance_matrix(n,m)
% 
%             % xl_nnhat=find(distance_matrix(user(n),:) < distance_matrix(user(n),n_hat));
%             % xl_mmhat=find(distance_matrix(user(m),:) < distance_matrix(user(m),m_hat));
%             % 
%             % for l=1:length(xl_nnhat)
%             %     plus_nhatmhat=plus_nhatmhat*(1-w(user(n),xl_nnhat(l)));
%             % end
%             % 
%             % for l=1:length(xl_mmhat)
%             %     plus_nhatmhat=plus_nhatmhat*(1-w(user(m),xl_mmhat(l)));
%             % end
% 
%             hnm=hnm+B_xn_xnhat(n,n_hat)*B_xn_xnhat(m,m_hat);
%         end
%     end
% end




less_position=double(xi_table{n,m} < xi_l);
B1=B_xn_xnhat(n,:);
B2=B_xn_xnhat(m,:);
hnm=B1 * less_position * B2';



end

