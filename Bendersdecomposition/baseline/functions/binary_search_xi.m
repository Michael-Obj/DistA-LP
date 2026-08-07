function xi = binary_search_xi(K, delta, ldx_left, ldx_right,w,n,m,distance_matrix,epsilon_nmw,EPSILON,B_xn_xnhat1,B_xn_xnhat2,Pr,xi_table_n_hat_m_hat)
    % Sort the elements in K in increasing order
    % K = sort(K(:));
    % HHH=[];
    % HHHH=[];
    % Binary search loop
    while ldx_left <= ldx_right
        ldx = floor((ldx_left + ldx_right) / 2);
        % Check the conditions for hn,m at K(ldx) and K(ldx - 1)

        tic;
        hnm_K(ldx)=B_xn_xnhat1*double(xi_table_n_hat_m_hat < K(ldx))*(B_xn_xnhat2)';
        time1=toc;
        tic;
        if ldx==1
            xi = K(ldx);
            return;
        end
        time2=toc;

        tic;
        hnm_K(ldx-1)=B_xn_xnhat1*double(xi_table_n_hat_m_hat < K(ldx-1))*(B_xn_xnhat2)';
        time3=toc;

        % HHH=[HHH,hnm_K(ldx)];
        % HHHH=[HHHH,hnm_K(ldx-1)];
        tic;
        if (hnm_K(ldx) >= (1 - delta)/Pr(n,m) && hnm_K(ldx - 1) <= (1 - delta)/Pr(n,m))||(ldx_right-ldx_left==1)
            xi = K(ldx);
            return;
        else
            % Search the right half if condition meets
            if hnm_K(ldx) < (1 - delta)/Pr(n,m)
                ldx_left = ldx;
            % Search the left half if condition meets
            elseif hnm_K(ldx - 1) >= (1 - delta)/Pr(n,m)
                ldx_right = ldx;
            end
        end
        time4=toc;
    end
end