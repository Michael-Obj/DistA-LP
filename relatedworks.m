%% Bubble chart for mDP related work
% X-axis = published year
% Bubble size = number of records; Color = type; Y-axis = avg utility loss.

clear; clc; close all;

%% ==== 1) Define your data (REPLACE with real numbers) ====
% Columns: {label, year, numRecords, avgUtilityLoss, type}
papers = { ...
    'Laplace (Andres13)',   2013,   5.0,   0.28,  'Predefined'; ...
    'LP (Bordenabe14)',     2014,   1.5,   0.18,  'Optimization'; ...
    'EM (Carvalho21)',      2021,   8.0,   0.22,  'Predefined'; ...
    'Decomp (Qiu22)',       2022,  30.0,   0.15,  'Optimization'; ...
    'Hybrid (Imola22)',     2022,  45.0,   0.17,  'Hybrid'; ...
    'Decomp (Qiu24)',       2024,  60.0,   0.14,  'Optimization'; ...
    'Hybrid (Qiu25)',       2025, 150.0,   0.12,  'Hybrid'; ...
    'Decomp (Liu25)',       2025, 120.0,   0.13,  'Optimization' ...
};

labels     = string(papers(:,1));
years      = cell2mat(papers(:,2));
numRecords = cell2mat(papers(:,3));   % scale as needed (e.g., thousands/millions)
avgLoss    = cell2mat(papers(:,4));
types      = string(papers(:,5));

%% ==== 2) Bubble size scaling ====
sizeMin = 40;   % points^2
sizeMax = 500;  % points^2
if range(numRecords) == 0
    sizes = repmat((sizeMin + sizeMax)/2, size(numRecords));
else
    sizes = sizeMin + (numRecords - min(numRecords)) ...
                  .* (sizeMax - sizeMin) / range(numRecords);
end

%% ==== 3) Color mapping by type ====
methodCats = categorical(types, {'Predefined','Optimization','Hybrid'});
catNames   = categories(methodCats);
colorMap = struct( ...
    'Predefined',   [0.10 0.48 0.84], ... % blue
    'Optimization', [0.20 0.62 0.20], ... % green
    'Hybrid',       [0.84 0.33 0.13] ...  % orange
);

%% Optional jitter so same-year points don't overlap horizontally
rng(7);  % for reproducibility
jitter = (rand(size(years)) - 0.5) * 0.20;  % ±0.1 years
xpos = years + jitter;

%% ==== 4) Plot ====
figure('Color','w'); hold on; box on;
h = gobjects(numel(catNames),1);
for ci = 1:numel(catNames)
    mask = (methodCats == catNames{ci});
    thisColor = colorMap.(char(catNames{ci}));
    h(ci) = scatter( ...
        xpos(mask), ...
        avgLoss(mask), ...
        sizes(mask), ...
        'MarkerFaceColor', thisColor, ...
        'MarkerEdgeColor', [0 0 0], ...
        'MarkerFaceAlpha', 0.75, ...
        'LineWidth', 0.5);
end

% Labels near bubbles
for i = 1:numel(labels)
    text(xpos(i), avgLoss(i), " " + labels(i), ...
        'FontSize', 8, 'VerticalAlignment', 'middle', 'Color', [0.15 0.15 0.15]);
end

% Axes & labels
xlabel('Publication year');
ylabel('Average utility loss (lower is better)');
title('mDP Mechanisms: Type, Scale (bubble size), Utility (y), and Year (x)');

% Year ticks as integers
xmin = floor(min(years) - 0.5);
xmax = ceil(max(years) + 0.5);
xlim([xmin xmax]);
set(gca, 'XTick', xmin:xmax, 'XGrid', 'on', 'YGrid', 'on', 'Layer', 'top');

% Legend (color = type)
legend(h, catNames, 'Location', 'northeastoutside', 'Title', 'Type');

% Optional size legend
sizeLegendVals = round(linspace(min(numRecords), max(numRecords), 3), 2);
sizeLegendPts  = sizeMin + (sizeLegendVals - min(numRecords)) ...
                         .* (sizeMax - sizeMin) / max(eps, range(numRecords));
p1 = scatter(nan, nan, sizeLegendPts(1), 'k', 'filled', 'MarkerFaceAlpha', 0.15);
p2 = scatter(nan, nan, sizeLegendPts(2), 'k', 'filled', 'MarkerFaceAlpha', 0.15);
p3 = scatter(nan, nan, sizeLegendPts(3), 'k', 'filled', 'MarkerFaceAlpha', 0.15);
legend([h(:); p1; p2; p3], [catNames; ...
    "Size: " + string(sizeLegendVals(1)), ...
    "Size: " + string(sizeLegendVals(2)), ...
    "Size: " + string(sizeLegendVals(3))], ...
    'Location','northeastoutside');

axis tight;
ylim([min(avgLoss) - 0.02*range(avgLoss), max(avgLoss) + 0.08*range(avgLoss)]);
set(gcf, 'Position', [100 100 980 430]);

% Save (vector + raster)
% exportgraphics(gcf, 'mdp_bubble_years.pdf', 'ContentType', 'vector');
% exportgraphics(gcf, 'mdp_bubble_years.png', 'Resolution', 300);
