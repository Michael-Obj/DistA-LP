function Dp = project_metric(D)
%PROJECT_METRIC  Enforce a proper metric on a square distance matrix.
% - Nonnegative
% - Symmetric
% - Zero diagonal
% - Triangle inequality via Floyd–Warshall (metric closure)
%
% D  : n×n real matrix (distances, can contain small negatives/NaNs/Infs)
% Dp : n×n projected metric distance matrix

    % --- basic checks ---
    if ~ismatrix(D) || size(D,1) ~= size(D,2)
        error('project_metric:NotSquare', 'Input must be a square matrix.');
    end
    n = size(D,1);

    % --- sanitize ---
    Dp = D;
    Dp(~isfinite(Dp)) = inf;           % treat NaN/Inf as very large distances
    Dp = max(Dp, 0);                   % nonnegative
    Dp = 0.5*(Dp + Dp.');              % symmetry
    Dp(1:n+1:end) = 0;                 % zero diagonal

    % --- Floyd–Warshall metric closure ---
    % Ensures: Dp(i,j) <= Dp(i,k) + Dp(k,j)  for all i,j,k
    for k = 1:n
        Dik = Dp(:,k);                 % n×1
        Dkj = Dp(k,:);                 % 1×n
        alt = Dik + Dkj;               % n×n (all i,j through k)
        % take elementwise min with current distances
        Dp = min(Dp, alt);
        Dp(1:n+1:end) = 0;            % keep diagonal exact zero
    end

    % final symmetry pass (numeric drift guard)
    Dp = 0.5*(Dp + Dp.');
end
