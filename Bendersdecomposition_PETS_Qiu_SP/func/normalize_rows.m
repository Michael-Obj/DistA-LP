function Mnorm = normalize_rows(M)
    mn = min(M, [], 1);
    mx = max(M, [], 1);
    Mnorm = (M - mn) ./ max(mx - mn, eps);
end