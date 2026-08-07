function epsilon_nmw = get_epsilon_nmw(q,all_target,distance_matrix,range_threshold)
epsilon_nmw=zeros(length(q),length(q));

num_nearest_n=[];
for n_hat_ind=1:length(all_target)
    n_hat=all_target(n_hat_ind);
    num_nearest_n=union(num_nearest_n,find(distance_matrix(n_hat,:)<0.5*range_threshold));       
end
all_epsilon_used=union(all_target,num_nearest_n);




%%
for i_cout=1:length(all_epsilon_used)
    i=all_epsilon_used(i_cout);
    for j_cout=1:length(all_epsilon_used)
        j=all_epsilon_used(j_cout);
        %epsilon_nmw(i,j)=rand;
        % beta_ij
        if i~=j
            epsilon_nmw_ij = 1;

            Rn_ij = Rn_for_mn(i,j,q);
            notin_Rn_ij = notin_Rn_for_mn(i,j,q);

            for l=length(Rn_ij)
                if l>0
                    epsilon_nmw_ij=epsilon_nmw_ij*q(i,Rn_ij(l))/q(j,Rn_ij(l));
                end
            end

            for l=length(notin_Rn_ij)
                %if (i~=notin_Rn_ij(l))&&(j~=notin_Rn_ij(l))
                if l>0
                    epsilon_nmw_ij=epsilon_nmw_ij*(1-q(i,notin_Rn_ij(l)))/(1-q(j,notin_Rn_ij(l)));
                end
                %end
            end
            epsilon_nmw(i,j)=log(epsilon_nmw_ij)/distance_matrix(i,j);
        end
    end
end
%epsilon_nmw=rand(length(q),length(q));



end
