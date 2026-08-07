load('location_data_r1_user2.mat')
lon=lon_sel;
lat=lat_sel;

%% Data: longitude lon and latitude lat (degrees)
% Assume lon and lat are existing vectors of equal length
valid = isfinite(lon) & isfinite(lat) & lon>=-180 & lon<=180 & lat>=-90 & lat<=90;
lon = lon(valid);  lat = lat(valid);

% %% Method A: simple scatter plot (quick distribution preview)
% figure; 
% scatter(lon, lat, 12, 'filled');   % Marker size is 12
% axis equal; grid on; box on;
% xlabel('Longitude'); ylabel('Latitude');
% title('Point Distribution (lon-lat)');
% % Optional: overlay coastlines (if the dataset is available)
% try
%     hold on; load coastlines; plot(coastlon, coastlat, 'k-'); hold off;
% end

%% Method B: geographic basemap (more intuitive; requires geographic axes)
figure; 
gx = geoaxes; 
geoscatter(lat, lon, 12, 'filled');     % Note the order: latitude comes first
geobasemap streets-light;               % Alternatives include 'satellite' and 'topographic'
title('Point Distribution on Basemap');
% Automatically set the view to the data range
latpad = range(lat)*0.05 + 0.1;  lonpad = range(lon)*0.05 + 0.1;
geolimits([min(lat)-latpad, max(lat)+latpad], [min(lon)-lonpad, max(lon)+lonpad]);

% %% Method C: density heatmap (shows clustering)
% figure;
% histogram2(lat, lon, [60 60], 'DisplayStyle','tile', 'ShowEmptyBins','off');
% colorbar; xlabel('Latitude'); ylabel('Longitude');
% title('Spatial Density (2D histogram)');
% set(gca,'YDir','normal');   % Use the conventional visual orientation for the axes
