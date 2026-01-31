function Btight = project_rectangular_triangle_tighten(A1, B, passes)
% Tighten rectangular block B (n×m) using triangle inequality with A1 (n×n).
% Enforces: B(i,j) <= min_k A1(i,k) + B(k,j)
% passes ~ 1–3 is usually enough.

    if nargin < 3 || isempty(passes), passes = 2; end
    A1 = max(A1,0); A1 = 0.5*(A1 + A1'); A1(1:size(A1,1)+1:end) = 0;
    Btight = max(B,0);

    [n,m] = size(Btight);
    for p = 1:passes
        for k = 1:n
            % For column j: new upper bound for all i is A1(:,k) + B(k,j)
            % Broadcast once per k:
            ub = A1(:,k) + Btight(k,:);   % n×m
            Btight = min(Btight, ub);
        end
    end
    Btight = max(Btight,0);
end
