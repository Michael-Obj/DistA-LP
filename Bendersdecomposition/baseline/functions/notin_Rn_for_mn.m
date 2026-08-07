function notin_Rn_for_mn = notin_Rn_for_mn(m,n,q)
notin_Rn_for_mn = find(q(m, :) <= q(n, :));
end
