data = [0.892853, 1.26164, 1.11716];
n = numel(data);

mean_val = mean(data);
std_val = std(data);

CI95 = 1.96 * std_val;
z_CI95 = 1.96 * std_val / sqrt(n);          % Approximate (large sample)
t_CI95 = tinv(0.975, n-1) * std_val / sqrt(n);  % Correct for small sample

fprintf('Mean = %.2f\n', mean_val);
% fprintf('Standard Deviation = %.2f\n', std_val);
% fprintf('Standard Deviation_95 = ±%.2f\n', CI95);
fprintf('95%% CI (approx, z) = ±%.2f\n', z_CI95);
% fprintf('95%% CI (exact, t) = ±%.2f\n', t_CI95);