function xi = linear_search_xi(K_sort, ldx_left, ldx_right,B_xn_xnhat1,B_xn_xnhat2,Pr,xi_table_n_hat_m_hat,row_idx_k_sort, col_idx_k_sort)
xi=0;    
hnm_K(1)=0;
ldx=ldx_left;
if isempty(row_idx_k_sort)
    1;
end
if isempty(col_idx_k_sort)
    2;
end
if isempty(K_sort)
    xi = 0;
    return;
end
    if xi_table_n_hat_m_hat(row_idx_k_sort(ldx),col_idx_k_sort(ldx))<=K_sort(1)
        hnm_K(1)=B_xn_xnhat1(1,row_idx_k_sort(ldx))*B_xn_xnhat2(1,col_idx_k_sort(ldx));
    end
    ldx_left=ldx_left+1;
    while ldx_left <= ldx_right
        ldx = ldx_left;
        hnm_K(ldx)=hnm_K(ldx-1)+B_xn_xnhat1(1,row_idx_k_sort(ldx))*B_xn_xnhat2(1,col_idx_k_sort(ldx));
        if (hnm_K(ldx) >= Pr && hnm_K(ldx - 1) <= Pr)||(ldx_right-ldx_left==1)
            xi = K_sort(ldx);
            return; 
        else
            ldx_left=ldx_left+1;
        end
        
    end
end

