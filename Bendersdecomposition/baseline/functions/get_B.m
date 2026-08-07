function B = get_B(w, distance_matrix, all_target)
B=zeros(length(distance_matrix),length(distance_matrix));
for n_row_ind=1:length(all_target)
    n_row=all_target(n_row_ind);
    for nhat_col=1:length(distance_matrix)
        xl_nnhat=find(distance_matrix(n_row,:) < distance_matrix(n_row,nhat_col));
        plus_nhatmhat=1;
        for l=1:length(xl_nnhat)
            plus_nhatmhat=plus_nhatmhat*(1-w(n_row,xl_nnhat(l)));
        end
        B(n_row,nhat_col)=plus_nhatmhat*w(n_row,nhat_col);
    end
end
end

