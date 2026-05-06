%% read_figure7_cc_losses.m
% Read loss_real_r_1.mat ... loss_real_r_11.mat from figure7/figure7_cc/

clear; clc;

NR_TAU = 100; 
NR_EPSILON = 10; 

baseDir = fullfile(pwd, "figure7", "figure7_ee"); % figure7_cc, figure7_ee
nFiles  = 15;
fileFmt = "loss_real_r_%d.mat";

if ~isfolder(baseDir)
    error("Folder not found: %s", baseDir);
end

lossVecs = cell(nFiles, 1);     % store vectors (works even if lengths differ)
found    = false(nFiles, 1);

for i = 1:nFiles
    fpath = fullfile(baseDir, sprintf(fileFmt, i));

    if ~isfile(fpath)
        warning("Missing file: %s", fpath);
        continue;
    end

    S = load(fpath);                 % struct with variables
    fn = fieldnames(S);

    if numel(fn) ~= 1
        error("File %s contains %d variables; expected exactly 1.", fpath, numel(fn));
    end

    v = S.(fn{1});                   % the vector

    if ~isvector(v)
        error("File %s: loaded variable is not a vector (size = %s).", ...
              fpath, mat2str(size(v)));
    end

    lossVecs{i} = v(:);              % store as column vector (consistent)
    found(i) = true;
end


fpath = fullfile(baseDir, 'num_group');
num_group = load(fpath);
num_group = num_group.num_group; 


loss = zeros(NR_TAU, NR_EPSILON); 
for i = 1:1:NR_TAU
    loss_inst = lossVecs{num_group(1, i)};
    loss(i, :) = loss_inst'; 
end


%% Plot the results

markers    = {'o','s','^','d','x','+','v','>','<','p','h','*','.'}; % enough variety
lineStyles = {'-','--',':','-.'};                                  % optional
tau = 0.01:0.01:1;

figure('Position', [200 200 900 700]);

% --- Use subplot (works in MATLAB 2024) ---
ax1 = subplot(2,1,1);  % top: multi-curve
ax2 = subplot(2,1,2);  % bottom: bar

% ===== Top plot (multiple curves) =====
axes(ax1); %#ok<LAXES>
hold on; grid on;

for r = 1:NR_EPSILON
    mk = markers{mod(r-1, numel(markers)) + 1};
    ls = lineStyles{mod(r-1, numel(lineStyles)) + 1};

    plot(tau, loss(:,r), ...          % loss is length(tau)-by-NR_EPSILON
        'LineStyle', ls, ...
        'Marker', mk, ...
        'LineWidth', 0.5, ...
        'MarkerSize', 5, ...
        'MarkerFaceColor', 'auto');
end

ax1.FontSize = 16;
ax1.LineWidth = 1.0;
ylabel(ax1, 'Utility loss', 'FontSize', 24);

legend(ax1, arrayfun(@(i) sprintf('\\epsilon = %d', i), 1:NR_EPSILON, 'UniformOutput', false), ...
       'Location', 'best', 'FontSize', 20);

hold off;

% ===== Bottom plot (bar) =====
axes(ax2); %#ok<LAXES>
bar(tau, num_group);                 % num_group should be same length as tau
ymax = max(num_group);
ylim(ax2, [0, 1.1*ymax]);   % 10% padding
grid on;

ax2.FontSize = 16;
ax2.LineWidth = 1.0;
ylabel(ax2, 'Number of groups', 'FontSize', 24);
xlabel(ax2, '\tau_{group}', 'FontSize', 24);

% ===== Resize subplot heights: bar is 0.5x as tall as top (2:1 ratio) =====
p1 = ax1.Position;   % [left bottom width height]
p2 = ax2.Position;

left  = min(p1(1), p2(1));
right = max(p1(1)+p1(3), p2(1)+p2(3));
width = right - left;

bottom = p2(2);
top    = p1(2) + p1(4);

gap = 0.04;                      % tweak if you want more/less space
availH = (top - bottom) - gap;

h2 = availH/3;                   % bottom height
h1 = 2*availH/3;                 % top height

ax2.Position = [left, bottom,          width, h2];
ax1.Position = [left, bottom+h2+gap,   width, h1];

% Optional: align x-limits
linkaxes([ax1 ax2], 'x');
