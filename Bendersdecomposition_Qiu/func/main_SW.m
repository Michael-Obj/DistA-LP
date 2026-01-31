% example script for using the OpenStreetMap functions
%
% use the example map.osm file in the release on github
%
% or
%
% download an OpenStreetMap XML Data file (extension .osm) from the
% OpenStreetMap website:
%   http://www.openstreetmap.org/
% after zooming in the area of interest and using the "Export" option to
% save it as an OpenStreetMap XML Data file, selecting this from the
% "Format to Export" options. The OSM XML is specified in:
%   http://wiki.openstreetmap.org/wiki/.osm
%
% See also PARSE_OPENSTREETMAP, PLOT_WAY, EXTRACT_CONNECTIVITY,
%          GET_UNIQUE_NODE_XY, ROUTE_PLANNER, PLOT_ROUTE, PLOT_NODES.
%
% 2010.11.25 (c) Ioannis Filippidis, jfilippidis@gmail.com

% references
% https://dl.acm.org/citation.cfm?id=102788 graph decomposition
 
%% READ FILES
% Save paths
addpath .\openstreetmap-master
savepath
% name file
% openstreetmap_filename = 'rome3.osm';
% openstreetmap_filename = 'cologne.osm';
openstreetmap_filename = 'maps/glassboro2.osm';
% http://kolntrace.project.citi-lab.fr/.
% map_img_filename = 'map.png'; % image file saved from online, if available

%% convert XML -> MATLAB struct
% convert the OpenStreetMap XML Data file donwloaded as map.osm
% to a MATLAB structure containing part of the information describing the
% transportation networsk
%% ----------------------- MAP INFORMATION ------------------------- 
% ****************** NOTICE: READ MAP INFORMATION ********************
[parsed_osm, osm_xml] = parse_openstreetmap(openstreetmap_filename);
% load('./data/osm_xml_rome.mat');
% load('./data/parsed_osm_rome.mat');

% Draw the map 
fig = figure;
ax = axes('Parent', fig);
hold(ax, 'on')
% plot the network, optionally a raster image can also be provided for the
% map under the vector graphics of the network
plot_way(ax, parsed_osm)
% plot_way(ax, parsed_osm, map_img_filename) % if you also have a raster image
% cologne_vehicletrace

%% PARAMETERS
delta = 0.01;       % interval length
ETA = 0.0015;        % privacy distance Default = 0.0015
epsilon = 800;         % Default = 1000
location_ID = 300;

%% DATA PREPROSESSING: connectivity, distance, intervals ... 
[connectivity_matrix, weight_matrix, intersection_node_indices, intervals] = extract_connectivity(parsed_osm, delta);
intersection_nodes = get_unique_node_xy(parsed_osm, intersection_node_indices);
%%%%%%%%%%%%%%%%%%%%%%% NOTICE: partition intervals %%%%%%%%%%%%%%%%%%%%%%%
% load('./data/intervals_rome.mat');
% intervals = interval_mod(intervals);

% try without the assumption of one-way roads
% start = 1; % node global index
% target = 9;
% dg = weight_matrix;
% dg = connectivity_matrix;
% dg = weight_matrix; 
% dg = min(weight_matrix, weight_matrix');
% dg = or(weight_matrix, weight_matrix.');
dg = or(connectivity_matrix, connectivity_matrix.'); % make symmetric
% [S,C] = graphconncomp(dg);

G = graph(dg);
bins = conncomp(G); 

node_idx = [];
for i = 1:1:size(bins, 2)
    if bins(i) == mode(bins)
        node_idx = [node_idx i];
    end
end

plot_nodes(ax, parsed_osm, node_idx);

%% Calculate the shortest distance between pairs
% distMatrix = zeros(size(intersection_node_indices, 2));
%%%%%%%%%%% NOTICE: calculate distance between intersection nodes %%%%%%%%%


%% Check Shangwen's errors ------------- 
for i = 1:1:size(distMatrix, 1)
    for j = 1:1:size(distMatrix, 2)
        [route, distMatrix(i, j)] = route_planner(dg, intersection_node_indices(i), intersection_node_indices(j));
%         if distMatrix(i, j) == inf
%             distMatrix(i, j) = max([normrnd(0.3, 0.05), 0.1]);
%         end
%       [i j distMatrix(i, j)]
    end
    i
end
% load('.\data\distMatrix_rome_adj.mat');

% load('.\data\shortest_distance_matrix.mat');
% connectivity_intersect_node = connectedIntersection(distMatrix); 
% load('.\data\shortest_distance_matrix_weight.mat')

%%%%%%%%%%%%%%% NOTICE: calculate distance between intervals %%%%%%%%%%%%%%
interval_distanceMatrix = interval_distancematrix(intervals, distMatrix, intersection_node_indices);
% load('.\data\interval_distanceMatrix_rome.mat'); 


%% ALGORITHM PART

% plan a route

%% try with the assumption of one-way roads (ways in OSM)
% start = 317; % node id
% target = 59;
% dg = connectivity_matrix; % directed graph
% dg = weight_matrix;
% [route, dist] = route_planner(dg, start, target);

%%%%%%%%%%%%%%% NOTICE: calculate cloaking matrix %%%%%%%%%%%%%%
% [cloakingmatrix, interval_eucldistanceMatrix, f_] = cloaking(interval_distanceMatrix, intervals, ETA, epsilon);
% load('.\data\cloakingmatrix_rome_adj.mat');

%%%%%%%%%%%%%%% NOTICE: calculate the lower bound %%%%%%%%%%%%%%%
% lowerboundmatrix = lowerbound(distMatrix, ETA, intervals, intersection_nodes, intersection_node_indices, epsilon);

%% ------------------------------- PLOT -----------------------------------
% plot_route(ax, route, parsed_osm)
% % only_nodes = 1:10:1000; % not all nodes, to reduce graphics memory & clutter
% % plot_nodes(ax, parsed_osm, only_nodes);
% % plot_nodes(ax, parsed_osm, intersection_node_indices);
% % show intersection nodes (unseful, but may result into a cluttered plot)
% plot_nodes(ax, parsed_osm, intersection_node_indices);
% hold on;
% plot_interval(ax, intervals, intersection_nodes, intersection_node_indices);
% plot_cloakingloc(cloakingmatrix, intervals, intersection_node_indices, intersection_nodes, location_ID, ETA); 
% 
% hold(ax, 'off');

% load('.\roma\vehicle_1000000_small.mat');


f_ = reshape(f_, size(cloakingmatrix));
for i = 1:1:size(vehicle, 2)
    
    EETD(i) = 0;
    privacy(i) = 0;
    for k = 1:1:size(vehicle(i).time, 2)
    % for k = 1:1:min([100 size(vehicle(i).time, 2)])
        % [i k size(vehicle(i).time, 2)]
        distance = zeros(size(intervals, 2), 1); 
        for j = 1:1:size(intervals, 2)
            distance_diff = intervals(j).loc - [vehicle(i).locy(k); vehicle(i).locx(k)]; 
            distance(j) = sqrt(sum(distance_diff.^2));
        end
        [distancemin min_idx] = min(distance);
        EETD(i) = EETD(i) + sum(f_(min_idx, :) .* cloakingmatrix(min_idx, :));
        privacy(i) = privacy(i) + sum(interval_eucldistanceMatrix(min_idx, :) .* cloakingmatrix(min_idx, :));
    end
    EETD(i) = EETD(i)/size(vehicle(i).locx,2);
    privacy(i) = privacy(i)/size(vehicle(i).locx,2);
    sizetrace(i) = size(vehicle(i).locx,2); 
    [i k]
end

EETD = EETD';
a = 0;

% clear;
% clc;

