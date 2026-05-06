addpath('./classes/Server/');
addpath('./classes/User/');
addpath('./classes/MasterProgram/');
addpath('./classes/Subproblem/');
addpath('./func/benchmarks/');
addpath('./func/benchmarks/randl/');
addpath('./func'); 
addpath('./func/read_files'); 
addpath('./func/haversine'); 

grid_size = 3; 
CRT_GRID_CELL_SIZE = 0.1; 

% rng("default")

%% Parameters 
parameters; 


% ----------------------------
% ----------------------------
% ROME DATASET
% ----------------------------
% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 42.1;
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.4; 
% TARGET_LON_MIN = 12.2; 
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401; 
% TARGET_LAT_MAX = 42.1; 
% TARGET_LAT_MIN = 41.901;

% TARGET_LON_MAX = 12.6; 
% TARGET_LON_MIN = 12.401;
% TARGET_LAT_MAX = 41.9; 
% TARGET_LAT_MIN = 41.701;

% TARGET_LON_MAX = 12.8; 
% TARGET_LON_MIN = 12.601; 
% TARGET_LAT_MAX = 42; 
% TARGET_LAT_MIN = 41.801;
% ----------------------------
% ----------------------------


% ----------------------------
% ----------------------------
% NYC DATASET
% ----------------------------
% TARGET_LON_MAX = -74; 
% TARGET_LON_MIN = -74.3; 
% TARGET_LAT_MAX = 40.65; 
% TARGET_LAT_MIN = 40.5;
% 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.95; 
% TARGET_LAT_MIN = 40.801;
% 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.8; 
% TARGET_LAT_MIN = 40.6501;
% 
% TARGET_LON_MAX = -73.7; 
% TARGET_LON_MIN = -74.01; 
% TARGET_LAT_MAX = 40.65; 
% TARGET_LAT_MIN = 40.5;
% ----------------------------
% ----------------------------


% ----------------------------
% ----------------------------
% LONDON DATASET
% ----------------------------
% TARGET_LON_MAX = -0.3; 
% TARGET_LON_MIN = -0.5; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;

% TARGET_LON_MAX = -0.1; 
% TARGET_LON_MIN = -0.301; 
% TARGET_LAT_MAX = 51.7; 
% TARGET_LAT_MIN = 51.501;

% TARGET_LON_MAX = -0.1; 
% TARGET_LON_MIN = -0.301; 
% TARGET_LAT_MAX = 51.5; 
% TARGET_LAT_MIN = 51.3;

% TARGET_LON_MAX = 0.1; 
% TARGET_LON_MIN = -0.101; 
% TARGET_LAT_MAX = 51.7; 
% TARGET_LAT_MIN = 51.501;

% TARGET_LON_MAX = 0.1; 
% TARGET_LON_MIN = -0.101; 
% TARGET_LAT_MAX = 51.5; 
% TARGET_LAT_MIN = 51.3;

% TARGET_LON_MAX = 0.3; 
% TARGET_LON_MIN = 0.101; 
% TARGET_LAT_MAX = 51.6; 
% TARGET_LAT_MIN = 51.4;
% ----------------------------
% ----------------------------

% env_parameters.longitude_min = TARGET_LON_MIN;
% env_parameters.longitude_max = TARGET_LON_MAX; 
% env_parameters.latitude_min = TARGET_LAT_MIN; 
% env_parameters.latitude_max = TARGET_LAT_MAX; 



Regions = [ ...
 %% ROME DATASET (FOR 2000 NODES)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.401,'lon_max',12.6,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.6,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.601,'lon_max',12.8,'lat_min',41.801,'lat_max',42.00) ];      

 %% ROME DATASET (4000 $ 6000 NODES)
 % struct('lon_min',12.2,'lon_max',12.4,'lat_min',41.701,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.59,'lat_min',41.901,'lat_max',42.10)
 % struct('lon_min',12.401,'lon_max',12.59,'lat_min',41.701,'lat_max',41.90)
 % struct('lon_min',12.5901,'lon_max',12.8,'lat_min',41.801,'lat_max',42.00) ];

 %% ROME DATASET (8000 $ 10000 NODES)
 % struct('lon_min',12.2,'lon_max',12.45,'lat_min',41.701,'lat_max',42.10)
 % struct('lon_min',12.4501,'lon_max',12.55,'lat_min',41.701,'lat_max',42.10)
 % struct('lon_min',12.5501,'lon_max',12.8,'lat_min',41.7501,'lat_max',42.00) ];


 
 %% NYC DATASET (2000, 4000 $ 6000 NODES)
 % struct('lon_min',-74.3,'lon_max',-74,'lat_min',40.5,'lat_max',40.65)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.801,'lat_max',40.95)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.6501,'lat_max',40.8)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.5,'lat_max',40.65) ]; 

 %% NYC DATASET (8000 $ 10000 NODES)
 % struct('lon_min',-74.3,'lon_max',-74,'lat_min',40.5,'lat_max',40.67)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.7801,'lat_max',40.95)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.6701,'lat_max',40.78)
 % struct('lon_min',-74.01,'lon_max',-73.7,'lat_min',40.5,'lat_max',40.67) ]; 



 %% LONDON DATASET
 struct('lon_min',-0.5,'lon_max',-0.3,'lat_min',51.4,'lat_max',51.6)
 struct('lon_min',-0.301,'lon_max',-0.1,'lat_min',51.501,'lat_max',51.7)
 struct('lon_min',-0.301,'lon_max',-0.1,'lat_min',51.3,'lat_max',51.5)
 % struct('lon_min',-0.101,'lon_max',0.1,'lat_min',51.501,'lat_max',51.7)
 struct('lon_min',-0.101,'lon_max',0.1,'lat_min',51.3,'lat_max',51.5)
 struct('lon_min',0.101,'lon_max',0.3,'lat_min',51.4,'lat_max',51.6) ];
R = numel(Regions);


LR_LOC_SIZE = 20;                                                           % The total number of locations
OBF_RANGE = 4.0;                                                            % The obfuscation range is considered as a circle, and OBF_RANGE is the radius
EXP_RANGE = 4.0;                                                            % The set of location not applying exponential mechanism is within a circle, of which the radius is EXP_RANGE. 
% OBF_RANGE = 20;                                                             % The obfuscation range is considered as a circle, and OBF_RANGE is the radius
% EXP_RANGE = 10;                                                             % The set of location not applying exponential mechanism is within a circle, of which the radius is EXP_RANGE. 
NEIGHBOR_THRESHOLD = 0.5;                                                   % The neighbor threshold eta
NR_DEST = 1;                                                                % The number of destinations (spatial tasks)
NR_USER = 10;                                                               % The number of users (agents)
LR_SAMPLE_SIZE = 100;                                                       % ??Michael
NR_LOC = 4;


% Initialize arrays for storing metrics across R regions
cost             = nan(1, R);   
cost_test        = nan(1, R);   
computation_time = nan(1, R);  
nr_violations    = nan(1, R);  
violation_mag    = nan(1, R);  

PL_DLR   = nan(R, NR_USER);   % Privacy Loss for Original Distance Matrix
Rel_DLR  = nan(R, NR_USER);   % Relative Frobenius error for Original Distance Matrix
Viol_DLR = nan(R, NR_USER);   % Violation Ratio for Original Distance Matrix
Rel_DLR2  = nan(R, NR_USER);  % Relative Frobenius error for Obfuscated Distance Matrix
Viol_DLR2 = nan(R, NR_USER);  % Violation Ratio for Obfuscated Distance Matrix
Rel_CRL = nan(R, NR_USER);    % Relative Frobenius error for Cost Matrix
% -------------------------------  
Rel_DLR_p   = nan(R, NR_USER);   
Viol_DLR_p  = nan(R, NR_USER);   
Rel_DLR2_p  = nan(R, NR_USER);  
Viol_DLR2_p = nan(R, NR_USER);     
Rel_CRL_p   = nan(R, NR_USER);    
% -------------------------------
Rel_DLR_r   = nan(R, NR_USER);   
Viol_DLR_r  = nan(R, NR_USER);  
Rel_DLR2_r  = nan(R, NR_USER);  
Viol_DLR2_r = nan(R, NR_USER);   
Rel_CRL_r   = nan(R, NR_USER);    
% -------------------------------
Rel_DLR_s   = nan(R, NR_USER);   
Viol_DLR_s  = nan(R, NR_USER);  
Rel_DLR2_s  = nan(R, NR_USER);  
Viol_DLR2_s = nan(R, NR_USER);     
Rel_CRL_s   = nan(R, NR_USER);    



baseSeed = 12345;                                   % any fixed number you like
stream   = RandStream('Threefry','Seed',baseSeed);  % or 'mrg32k3a' if you prefer
RandStream.setGlobalStream(stream);                 % make it the global RNG


%% SVD APPROXIMATION

%% 2000 NODES
% budget_assigned_rome_svd = [2.18546+2.81987+2.69885, 2.53849+1.99343+3.83922, 2.35294+1.94143+2.51932, 1.826+1.56157+2.81624, 2.9957+1.84421+3.62236;
%                             1.53118+1.90233+1.89665, 1.77391+1.37776+2.70288, 1.6383+1.3368+1.74504, 1.2675+1.07978+1.96616, 2.10123+1.27424+2.53739;
%                             0.878507+0.999485+1.09303, 1.01561+0.764918+1.58353, 0.931265+0.739451+0.979383, 0.712727+0.602883+1.11846, 1.21244+0.710438+1.45168];


% budget_assigned_nyc_svd = [2.20619+1.41263+1.78887, 3.08017+1.69431+3.56499, 2.03086+1.4251+1.75134, 2.80096+1.89003+2.99041;
%                             1.53265+0.985633+1.26821, 2.16351+1.17596+2.49272, 1.43437+0.97867+1.22917, 1.95749+1.32459+2.1238;
%                             0.861694+0.560251+0.737302, 1.24698+0.671905+1.43023, 0.834931+0.540999+0.705903, 1.11796+0.759769+1.24571]; 

% budget_assigned_london_svd = [2.07727+1.47116+1.39154, 2.4405+1.31729+1.13751, 2.03756+1.30895+1.29848, 2.10788+1.67461+2.07272, 2.25325+2.49798+2.14005;
%                               1.43576+1.0256+0.967367, 1.71806+0.911152+0.778774, 1.41762+0.910678+0.896873, 1.47476+1.17042+1.45152, 1.56803+1.70783+1.49715;
%                               0.804163+0.581742+0.540023, 0.994337+0.502642+0.431062, 0.805045+0.514847+0.491581, 0.846014+0.667039+0.830238, 0.888136+0.916871+0.85487];




%% 4000 NODES
% budget_assigned_rome_svd_10 = [1.62388+0.845337+1.66844, 1.8756+1.27814+1.67027, 2.02693+1.14313+1.24099, 2.48513+2.1237+1.90402];
% budget_assigned_rome_svd_7 = [1.13521+0.583166+1.17936, 1.31269+0.890921+1.16058, 1.40984+0.768284+0.840288, 1.74501+1.4703+1.28759];
% budget_assigned_rome_svd_4 = [0.647849+0.343846+0.663307, 0.750463+0.506235+0.655254, 0.79739+0.443967+0.453896, 1.00592+0.824146+0.705942];

% budget_assigned_nyc_svd = [ 2.45931+1.50217+1.90259, 3.15822+2.17249+2.60937, 1.88444+1.26694+1.35802, 2.27408+1.80297+2.49661;
%                             1.70038+1.0545+1.34386, 2.17397+1.46558+1.90503, 1.31407+0.876652+0.955528, 1.58301+1.25729+1.7492;
%                             0.949427+0.604199+0.775912, 1.20172+0.793139+1.13732, 0.748963+0.487795+0.545097, 0.894131+0.710801+1.00142];

 
% budget_assigned_london_svd = [ 2.11182+1.44095+1.303, 2.37419+1.27126+1.46142, 1.8639+1.519+1.60196, 2.42572+1.761+4.38661, 2.13236+1.4729+2.48193, 1.81309+1.5008+3.17953;
%                                 1.48405+1.00063+0.902679, 1.66309+0.880615+1.02625, 1.29909+1.05727+1.11383, 1.673+1.20155+3.15689, 1.49009+1.02829+1.71841, 1.261+1.04381+2.30867;
%                                 0.855449+0.5665+0.525164, 0.953922+0.492001+0.586758, 0.73848+0.599278+0.629809, 0.932873+0.64288+1.74992, 0.849559+0.577794+0.965131, 0.712613+0.590561+1.27821];


%% 6000 NODES
% budget_assigned_rome_svd = [1.49209+0.855817+1.57077, 2.21483+2.7719+1.84124, 1.81025+1.50167+2.80508, 2.37343+1.53947+1.62647;
%                             1.04373+0.584464+1.09165, 1.55468+1.93691+1.2888, 1.26668+1.04692+1.96345, 1.66851+1.06931+1.13999;
%                             0.59924+0.317372+0.612933, 0.892172+1.10497+0.736384, 0.725394+0.593761+1.12183, 0.962795+0.599714+0.652864];

% budget_assigned_nyc_svd = [ 2.14486+1.49932+2.08785, 3.0748+2.05622+3.1526, 1.76184+1.14468+1.47996, 2.46916+1.85382+2.42909;
%                             1.50525+1.04001+1.46399, 2.16481+1.43805+2.23435, 1.22985+0.800573+1.03492, 1.73347+1.30219+1.745;
%                             0.86622+0.581582+0.838164, 1.25281+0.82066+1.18181, 0.70517+0.455312+0.592686, 1.00055+0.7515+1.03689];

% budget_assigned_london_svd = [2.21878+1.4019+1.47554, 2.42005+1.32984+1.57341, 2.24172+1.53018+1.5951, 2.31122+1.79202+5.13744, 1.85732+1.33001+1.33056, 2.27615+1.26644+1.76592;
%                               1.53876+0.959374+1.03089, 1.68874+0.91019+1.11015, 1.54889+1.0684+1.11282, 1.60936+1.23365+3.51609, 1.29549+0.928208+0.938619, 1.57825+0.875976+1.20024;
%                               0.865705+0.523138+0.587715, 0.962239+0.514493+0.643542, 0.863752+0.597545+0.641775, 0.916812+0.680442+2.0072, 0.735414+0.527751+0.529931, 0.888763+0.488658+0.634356];


%% 8000 NODES


budget_assigned_london_svd = [2.55543+1.51772+2.1159	2.22946+1.66026+1.36408	1.96005+1.23427+1.35916	2.47258+2.35932+2.07946	2.25706+1.81665+1.41181	2.52219+2.13777+3.40893;
                              1.77623+1.05807+1.48815	1.55195+1.1418+0.954854	1.37311+0.84555+0.948043	1.72342+1.60311+1.47309	1.58982+1.25001+0.987592	1.76897+1.4639+2.38635; 
                              1.00547+0.600049+0.858469	0.879415+0.626917+0.533691	0.788681+0.477735+0.53871	0.986567+0.877025+0.862396	0.923459+0.686083+0.553896	1.01301+0.794106+1.36408];



%% 10000 NODES
budget_assigned_rome_svd = [1.32127+0.816568+2.23292	1.41095+2.024+3.17445	1.69799+2.00649+2.78936;
                            0.927028+0.569959+1.56286	0.9809+1.40295+2.22484	1.19234+1.38075+1.95344;
                            0.53374+0.324043+0.892853	0.555674+0.781799+1.26164	0.686444+0.761744+1.11716];

%% GAUSSIAN APPROXIMATION
%% 2000 NODES
% budget_assigned_rome_svd_4 = [0.878507+0.999485+1.09303, 1.01561+0.764918+1.58353, 0.931265+0.739451+0.979383, 0.712727+0.602883+1.11846, 1.21244+0.710438+1.45168]; 
% budget_assigned_nyc_svd_4 = [0.861694+0.560251+0.737302, 1.24698+0.671905+1.43023, 0.834931+0.540999+0.705903, 1.11796+0.759769+1.24571]; 
% budget_assigned_london_svd_4 = [0.804163+0.581742+0.540023, 0.994337+0.502642+0.431062, 0.805045+0.514847+0.491581, 1.09985+0.665564+2.35574, 0.846014+0.667039+0.830238, 0.888136+0.916871+0.85487]; 

% budget_assigned_rome_svd_7 = [1.53118+1.90233+1.89665, 1.77391+1.37776+2.70288, 1.6383+1.3368+1.74504, 1.2675+1.07978+1.96616, 2.10123+1.27424+2.53739]; 
% budget_assigned_nyc_svd_7 = [1.53265+0.985633+1.26821, 2.16351+1.17596+2.49272, 1.43437+0.97867+1.22917, 1.95749+1.32459+2.1238]; 
% budget_assigned_london_svd_7 = [1.43576+1.0256+0.967367, 1.71806+0.911152+0.778774, 1.41762+0.910678+0.896873, 1.88969+1.21788+4.12336, 1.47476+1.17042+1.45152, 1.56803+1.70783+1.49715]; 

% budget_assigned_rome_svd_10 = [2.18546+2.81987+2.69885, 2.53849+1.99343+3.83922, 2.35294+1.94143+2.51932, 1.826+1.56157+2.81624, 2.9957+1.84421+3.62236]; 
% budget_assigned_nyc_svd_10 = [2.20619+1.41263+1.78887, 3.08017+1.69431+3.56499, 2.03086+1.4251+1.75134, 2.80096+1.89003+2.99041]; 
% budget_assigned_london_svd_10 = [2.07727+1.47116+1.39154, 2.4405+1.31729+1.13751, 2.03756+1.30895+1.29848, 2.67986+1.79413+5.88626, 2.10788+1.67461+2.07272, 2.25325+2.49798+2.14005]; 


% budget_assigned_nyc_svd = [1.63828+2.02546+1.30905, 1.39846+0.784722+1.31867, 0.835727+0.831656+0.50948, 1.45594+1.62417+2.28182;
%                             1.14502+1.43213+0.915083, 0.955607+0.545514+0.921849, 0.569312+0.5622+0.358812, 1.01777+1.13048+1.59537;
%                             0.652537+0.766714+0.521796, 0.522822+0.304622+0.524984, 0.316527+0.305862+0.211581, 0.583017+0.646035+0.909545]; 

% budget_assigned_london_svd = [1.59352+1.12965+5.52618, 1.37703+2.60843+1.19561, 2.53558+2.85682+1.46094, 3.82615+1.77105+3.7712, 3.12055+0.531954+0.768607, 2.35696+1.14288+2.62506;
%                               1.13096+0.790077+3.87388, 0.963284+1.82512+0.835761, 1.77334+1.99835+1.0215, 2.69911+1.23871+2.5998, 2.21518+0.372158+0.519553, 1.63557+0.788885+1.83772;
%                               0.602435+0.456466+2.13018, 0.550236+1.04171+0.476318, 1.01176+1.1411+0.58256, 1.47918+0.706301+1.48161, 1.21191+0.212461+0.281167, 0.925412+0.456227+1.0471];


%% 4000 nodes
% budget_assigned_rome_svd_4 = [0.509496+0.346576+0.339959, 0.873798+0.389929+0.33255, 0.468704+0.270966+0.76436, 0.740074+0.467947+0.919309];
% budget_assigned_nyc_svd_4 = [0.706044+1.07503+1.12086, 0.432367+0.956446+0.671128, 0.512108+0.462501+0.24248, 0.494+0.400418+0.55175];


% budget_assigned_rome_svd_7 = [0.892032+0.612807+0.578182, 1.53004+0.699578+0.600965, 0.911458+0.526901+1.32253, 1.31123+0.893162+1.61512];
% budget_assigned_nyc_svd_7 = [1.27595+1.91487+1.96406, 0.794771+1.60774+1.17699, 0.898771+0.787024+0.443313, 0.882418+0.76803+0.961244]; 


% budget_assigned_rome_svd_10 = [1.27529+0.889194+0.837009, 2.18753+1.01664+0.878486, 1.37845+0.793786+1.89827, 1.86787+1.36884+2.30422]; 
% budget_assigned_nyc_svd_10 = [1.87064+2.72368+2.80797, 1.16854+2.43557+1.68631, 1.28454+1.1162+0.654628, 1.30812+1.14853+1.37032]; 

% budget_assigned_rome_svd = [ 1.27529+0.889194+0.837009, 2.18753+1.01664+0.878486, 1.37845+0.793786+1.89827, 1.86787+1.36884+2.30422;
%                              0.892032+0.612807+0.578182, 1.53004+0.699578+0.600965, 0.911458+0.526901+1.32253, 1.31123+0.893162+1.61512; 
%                              0.509496+0.346576+0.339959, 0.873798+0.389929+0.33255, 0.468704+0.270966+0.76436, 0.740074+0.467947+0.919309];


% budget_assigned_nyc_svd = [ 1.06852+1.58518+1.13902, 1.52506+1.77438+1.49216, 0.784169+0.813363+0.733969, 1.63486+1.43123+0.838577; 
%                             0.747796+1.11259+0.780784, 1.02639+1.19664+1.01035, 0.546918+0.560276+0.491643, 1.12079+1.00548+0.585682;
%                             0.427242+0.657618+0.435168, 0.538095+0.642948+0.546101, 0.311479+0.319862+0.256013, 0.644861+0.577851+0.334668];


% budget_assigned_london_svd = [1.42452+0.786515+0.996493, 3.28097+1.75099+1.93551, 2.78632+1.26294+0.841731, 6.62382+1.38448+0.795869, 1.08626+0.591008+0.859319, 4.17165+1.68721+1.12638;
%                               0.996291+0.552909+0.692738, 2.29579+1.2233+1.35403, 1.92006+0.862436+0.584477, 4.63378+0.945309+0.556624, 0.753427+0.407751+0.581795, 2.91936+1.17419+0.79518; 
%                               0.568786+0.31807+0.395118, 1.31136+0.704741+0.773127, 1.09+0.478878+0.325968, 2.64783+0.51991+0.317405, 0.426456+0.227827+0.317356, 1.66767+0.668454+0.457647];



%% 6000 nodes

% budget_assigned_rome_svd_4 = [0.321941+0.325364+0.616142, 0.569216+0.503903+0.355851, 3.20037+1.74876+0.420891, 0.592949+0.743646+2.00007];
% budget_assigned_nyc_svd_4 = [0.706044+1.07503+1.12086, 0.432367+0.956446+0.671128, 0.512108+0.462501+0.24248, 0.494+0.400418+0.55175];

% budget_assigned_rome_svd_7 = [0.57525+0.577833+1.0996, 0.976477+0.878648+0.624467, 5.93849+3.06033+0.764863, 1.04423+1.30551+3.50387];
% budget_assigned_nyc_svd_7 = [1.27595+1.91487+1.96406,  0.794771+1.60774+1.17699, 0.898771+0.787024+0.443313, 0.882418+0.76803+0.961244];


% budget_assigned_rome_svd_10 = [0.850963+0.836974+1.61656, 1.38872+1.2556+0.893898, 8.49798+4.37214+1.12143, 1.49435+1.86772+5.00787]; 
% budget_assigned_nyc_svd_10 = [1.87064+2.72368+2.80797, 1.16854+2.43557+1.68631, 1.28454+1.1162+0.654628,  1.30812+1.14853+1.37032];


% budget_assigned_nyc_svd = [ 1.87064+2.72368+2.80797, 1.16854+2.43557+1.68631, 1.28454+1.1162+0.654628, 1.30812+1.14853+1.37032;
%                             1.27595+1.91487+1.96406, 0.794771+1.60774+1.17699, 0.898771+0.787024+0.443313, 0.882418+0.76803+0.961244;
%                             0.706044+1.07503+1.12086, 0.432367+0.956446+0.671128, 0.512108+0.462501+0.24248, 0.494+0.400418+0.55175];

% budget_assigned_london_svd = [1.58657+2.30444+1.28489, 2.02598+1.18787+1.05494, 1.03573+1.76098+0.72468, 1.49762+1.41751+1.20609, 1.14068+1.19711+1.68871, 1.18526+2.91106+4.63768;
%                               1.10935+1.60744+0.89829, 1.44528+0.819521+0.717178, 0.720514+1.22815+0.499869, 1.04736+0.99137+0.784081, 0.770904+0.837273+1.17745, 0.822763+2.03748+3.24567
%                               0.62936+0.917905+0.512156, 0.797236+0.464261+0.392907, 0.411682+0.710831+0.279278, 0.598106+0.565829+0.388221, 0.419036+0.478251+0.671679, 0.464172+1.16438+1.85147];


% budget_assigned_rome_svd = [  0.850963+0.836974+1.61656, 1.38872+1.2556+0.893898, 8.49798+4.37214+1.12143, 1.49435+1.86772+5.00787;
%                                 0.57525+0.577833+1.0996, 0.976477+0.878648+0.624467, 5.93849+3.06033+0.764863, 1.04423+1.30551+3.50387;
%                                 0.321941+0.325364+0.616142, 0.569216+0.503903+0.355851, 3.20037+1.74876+0.420891, 0.592949+0.743646+2.00007];



%% Select the algorithm here
approx_matrix = 1;  % 1: approximated matrix is used; 0: approximated matrix is not used
epsilon_value = [10, 7, 4]; 
budget_assigned = budget_assigned_london_svd;
for ep_idx = 1:1:3
    % budget_assigned = budget_assigned_rome_svd_10;

    EPSILON = epsilon_value(1, ep_idx); 
    
    for r =  1:R      % London TR
    % for r = 1:R
        if approx_matrix == 1
            env_parameters.EPSILON = EPSILON - budget_assigned(ep_idx, r); 
        else
            env_parameters.EPSILON = EPSILON; 
        end
        % env_parameters.EPSILON = 4; 
        % --- set region bounds for this run ---
        env_parameters.longitude_min = Regions(r).lon_min;
        env_parameters.longitude_max = Regions(r).lon_max;
        env_parameters.latitude_min  = Regions(r).lat_min;
        env_parameters.latitude_max  = Regions(r).lat_max;
    
        env_parameters.nr_loc_selected = 500; 
       
        env_parameters.nr_loc_selected = NR_LOC*2000; 
        
        
        %% Initialization
        env_parameters = readCityMapInfo(env_parameters);                           % Create the road map information of the target region: Rome, Italy
        % env_parameters = readGridMapInfo(env_parameters);                         % Create the road map information of the target region: Rome, Italy
        env_parameters.GAMMA = 1000; 
        env_parameters.NEIGHBOR_THRESHOLD = 50;
     
    
        %% Create the server
        server = Server(NR_DEST, EXP_RANGE, CRT_GRID_CELL_SIZE);                    % Create the server
    
          
        %% Create the users        
        for m = 1:1:NR_USER
    
            stream.Substream = (r-1)*NR_USER + m;          % 1..(R*NR_USER)
            RandStream.setGlobalStream(stream);
    
            idx_selected = randperm(size(env_parameters.node_target, 2), env_parameters.nr_loc_selected); 
            env_parameters.longitude_selected = env_parameters.longitude(idx_selected); 
            env_parameters.latitude_selected = env_parameters.latitude(idx_selected);
            env_parameters.node_target_selected = env_parameters.node_target(idx_selected); 
            env_parameters.G_mDP = mDP_graph_creator(env_parameters);
    
            user(m, 1) = User(m, LR_LOC_SIZE, OBF_RANGE, NEIGHBOR_THRESHOLD, env_parameters);               % Create users
            user(m, 1) = user(m, 1).initialization(env_parameters);                                         % Initialize the properties of the user, including the local relevant locations, distance matrices, obfuscated location IDs, and the cost matrix
            
            if approx_matrix == 1
                user(m, 1) = user(m, 1).apply_svd();
                % user(m, 1) = user(m, 1).apply_gaussian(); 

            end
    
            lon_sel    = env_parameters.longitude_selected;
            lat_sel    = env_parameters.latitude_selected;
            node_tar   = env_parameters.node_target_selected;
            LR_ID      = user(m,1).LR_loc_ID;
            obf_ID     = user(m,1).obf_loc_ID;
            cost_matrix= user(m,1).cost_matrix_RL;
    
            % unique filename per region/user to avoid overwrite
            fname = sprintf('location_data_r%d_user%d.mat', r, m);
            save(fname, 'lon_sel','lat_sel','node_tar','LR_ID','obf_ID','cost_matrix','-v7.3');    
        end 
            
        server = server.destination_identifier(env_parameters); 
        % server = server.cr_table_cal(env_parameters);                               % Create the cost reference table
        % indist_set(grid_size, :) = threatByCostMatrix(server.cr_table, CRT_GRID_CELL_SIZE, 1); 
        server.exp_range = EXP_RANGE; 
    
        server = server.initialization(user);                                       % Create the destinations in the target region
    
        % for m = 1:1:NR_USER
        %     user(m, 1) = user(m, 1).cost_matrix_cal(server.cr_table, env_parameters);
        % end
    
    
        %% Local relevant geo-obfuscation algorithm
        tic;
        server = server.geo_obfuscation_initialization(user, env_parameters);        
        [server, user, nr_iterations, cost(ep_idx, r), cost_test(ep_idx, r)] = server.geo_obfuscation_generator(user, env_parameters);    % Generate the geo-obfuscation matrices 
        cost(ep_idx, r) = cost(ep_idx, r)/20;
        cost_test(ep_idx, r) = cost_test(ep_idx, r)/20;
    
    
        computation_time(ep_idx, r) = toc; 
        [nr_violations(ep_idx, r), violation_mag(ep_idx, r)]= GeoInd_violation_cnt(user, env_parameters); 
    
        for m = 1:NR_USER
            u = user(m); % (NR_USER=10 in your script)
    
            if ~isempty(u.distance_matrix_LR_recovered)
                % [~, ~, PL_max_1] = compute_log_posterior(u.fitted_best_params_1, u.n_fitted_best_params_1, u.distance_matrix, size(u.distance_matrix_LR,1), 1/u.epsilon, size(u.distance_matrix_LR,1)); 
                % PL_DLR(r) = PL_max_1; 
                Rel_DLR(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered);
                Viol_DLR(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered);
            end    
            if ~isempty(u.distance_matrix_LR2obf_recovered)
                Rel_DLR2(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered);
                Viol_DLR2(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered);
            end    
            if ~isempty(u.cost_matrix_RL_recovered)
                Rel_CRL(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered);
            end
            % -------------------------------------------------------------------------------------------------------------
            if ~isempty(u.distance_matrix_LR_recovered_p)    
                Rel_DLR_p(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_p);
                Viol_DLR_p(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_p);
            end   
            if ~isempty(u.distance_matrix_LR2obf_recovered_p)
                Rel_DLR2_p(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_p);
                Viol_DLR2_p(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_p);
            end   
            if ~isempty(u.cost_matrix_RL_recovered_p)
                Rel_CRL_p(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_p);
            end
            % ------------------------------------------------------------------------------------------------------------
            if ~isempty(u.distance_matrix_LR_recovered_r)
                Rel_DLR_r(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_r);
                Viol_DLR_r(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_r);
            end    
            if ~isempty(u.distance_matrix_LR2obf_recovered_r)
                Rel_DLR2_r(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_r);
                Viol_DLR2_r(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_r);
            end    
            if ~isempty(u.cost_matrix_RL_recovered_r)
                Rel_CRL_r(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_r);
            end
            % ------------------------------------------------------------------------------------------------------------
            if ~isempty(u.distance_matrix_LR_recovered_s)
                Rel_DLR_s(r, m)  =  relative_error( u.distance_matrix_LR, u.distance_matrix_LR_recovered_s);
                Viol_DLR_s(r, m) =  violation_ratio(u.distance_matrix_LR, u.distance_matrix_LR_recovered_s);
            end    
            if ~isempty(u.distance_matrix_LR2obf_recovered_s)
                Rel_DLR2_s(r, m)  = relative_error( u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_s);
                Viol_DLR2_s(r, m) = violation_ratio(u.distance_matrix_LR2obf, u.distance_matrix_LR2obf_recovered_s);
            end    
            if ~isempty(u.cost_matrix_RL_recovered_s)
                Rel_CRL_s(r, m) =   relative_error(u.cost_matrix_RL, u.cost_matrix_RL_recovered_s);
            end
        end
    
        % for m = 1:NR_USER
        %     % make region-specific subfolders
        %     outG = sprintf('london_fig_surfaces_6000/r%02d/g', r);
        %     outP = sprintf('london_fig_surfaces_6000/r%02d/p', r);
        %     outR = sprintf('london_fig_surfaces_6000/r%02d/rbf', r);
        %     outS = sprintf('london_fig_surfaces_6000/r%02d/svd', r);
        % 
        %     if ~exist(outG,'dir'), mkdir(outG); end
        %     if ~exist(outP,'dir'), mkdir(outP); end
        %     if ~exist(outR,'dir'), mkdir(outR); end
        %     if ~exist(outS,'dir'), mkdir(outS); end
        % 
        %     % pass a full, user-specific stem to avoid collisions
        %     stemG = sprintf('%s/fig_surfaces_g_r%02d_u%02d', outG, r, m);
        %     stemP = sprintf('%s/fig_surfaces_p_r%02d_u%02d', outP, r, m);
        %     stemR = sprintf('%s/fig_surfaces_r_r%02d_u%02d', outR, r, m);
        %     stemS = sprintf('%s/fig_surfaces_s_r%02d_u%02d', outS, r, m);
        % 
        %     plot_all_surfaces_for_user_g(user(m), stemG);
        %     plot_all_surfaces_for_user_p(user(m), stemP);
        %     plot_all_surfaces_for_user_r(user(m), stemR);
        %     plot_all_surfaces_for_user_s(user(m), stemS);
        % end
    end
end

save("cost_test.mat", "cost_test"); 
save("nr_violations.mat", "nr_violations");
save("violation_mag.mat", "violation_mag");
save("computation_time.mat", "computation_time"); 

% mean_cost_test        = mean(cost_test, 'omitnan');          ci95_cost_test        = 1.96 * std(cost_test, 0, 'omitnan')/sqrt(R); %/ sqrt(numel(cost_test));
% mean_nr_violations    = mean(nr_violations, 'omitnan');      ci95_nr_violations    = 1.96 * std(nr_violations, 0, 'omitnan')/sqrt(R); % / sqrt(numel(nr_violations));
% mean_violation_mag    = mean(violation_mag, 'omitnan');      ci95_violation_mag    = 1.96 * std(violation_mag, 0, 'omitnan')/sqrt(R); % / sqrt(numel(violation_mag));
% mean_computation_time = mean(computation_time, 'omitnan');   ci95_computation_time = 1.96 * std(computation_time, 0, 'omitnan')/sqrt(R); % / sqrt(numel(computation_time));
% 
% fprintf('Cost Test    - Mean: %.4f, Std: %.4f\n', mean_cost_test, ci95_cost_test);
% fprintf('Violations   - Mean: %.4f, Std: %.4f\n', mean_nr_violations, ci95_nr_violations);
% fprintf('Viol_mag     - Mean: %.4f, Std: %.4f\n', mean_violation_mag, ci95_violation_mag);
% fprintf('Time (sec)   - Mean: %.4f, Std: %.4f\n', mean_computation_time, ci95_computation_time);
% 
% save('metrics_summary.mat', ...
%     'mean_cost_test','ci95_cost_test', ...
%     'mean_nr_violations','ci95_nr_violations', ...
%     'mean_violation_mag','ci95_violation_mag', ...
%     'mean_computation_time','ci95_computation_time');




% Collapse user dimension to region means (omit NaNs)
Rel_DLR_mean_by_region   = mean(Rel_DLR,  2, 'omitnan');
Viol_DLR_mean_by_region  = mean(Viol_DLR, 2, 'omitnan');
Rel_DLR2_mean_by_region  = mean(Rel_DLR2,  2, 'omitnan');
Viol_DLR2_mean_by_region = mean(Viol_DLR2, 2, 'omitnan');
Rel_CRL_mean_by_region   = mean(Rel_CRL, 2, 'omitnan');

% Then your overall means/stds across regions:
Mean_Rel_DLR  = mean(Rel_DLR_mean_by_region,  'omitnan');   CI95_Rel_DLR  = 1.96 * std(Rel_DLR_mean_by_region,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region));
Mean_Vio_DLR  = mean(Viol_DLR_mean_by_region, 'omitnan');   CI95_Vio_DLR  = 1.96 * std(Viol_DLR_mean_by_region, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region));
Mean_Rel_DLR2 = mean(Rel_DLR2_mean_by_region, 'omitnan');   CI95_Rel_DLR2 = 1.96 * std(Rel_DLR2_mean_by_region, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region));
Mean_Vio_DLR2 = mean(Viol_DLR2_mean_by_region,'omitnan');   CI95_Vio_DLR2 = 1.96 * std(Viol_DLR2_mean_by_region,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region));
Mean_Rel_CRL  = mean(Rel_CRL_mean_by_region,  'omitnan');   CI95_Rel_CRL  = 1.96 * std(Rel_CRL_mean_by_region,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region));

Summary = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR;  Mean_Vio_DLR;  Mean_Rel_DLR2;  Mean_Vio_DLR2;  Mean_Rel_CRL], ...
    [CI95_Rel_DLR;    CI95_Vio_DLR;    CI95_Rel_DLR2;    CI95_Vio_DLR2;    CI95_Rel_CRL], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);

disp(Summary);
writetable(Summary, 'Summary_Gaussian.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_p   = mean(Rel_DLR_p,  2, 'omitnan');
Viol_DLR_mean_by_region_p  = mean(Viol_DLR_p, 2, 'omitnan');
Rel_DLR2_mean_by_region_p  = mean(Rel_DLR2_p,  2, 'omitnan');
Viol_DLR2_mean_by_region_p = mean(Viol_DLR2_p, 2, 'omitnan');
Rel_CRL_mean_by_region_p   = mean(Rel_CRL_p, 2, 'omitnan');

Mean_Rel_DLR_p  = mean(Rel_DLR_mean_by_region_p,  'omitnan');   CI95_Rel_DLR_p  = 1.96 * std(Rel_DLR_mean_by_region_p,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_p));
Mean_Vio_DLR_p  = mean(Viol_DLR_mean_by_region_p, 'omitnan');   CI95_Vio_DLR_p  = 1.96 * std(Viol_DLR_mean_by_region_p, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_p));
Mean_Rel_DLR2_p = mean(Rel_DLR2_mean_by_region_p, 'omitnan');   CI95_Rel_DLR2_p = 1.96 * std(Rel_DLR2_mean_by_region_p, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_p));
Mean_Vio_DLR2_p = mean(Viol_DLR2_mean_by_region_p,'omitnan');   CI95_Vio_DLR2_p = 1.96 * std(Viol_DLR2_mean_by_region_p,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_p));
Mean_Rel_CRL_p  = mean(Rel_CRL_mean_by_region_p,  'omitnan');   CI95_Rel_CRL_p  = 1.96 * std(Rel_CRL_mean_by_region_p,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_p));

Summary_p = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_p;  Mean_Vio_DLR_p;  Mean_Rel_DLR2_p;  Mean_Vio_DLR2_p;  Mean_Rel_CRL_p], ...
    [CI95_Rel_DLR_p;    CI95_Vio_DLR_p;    CI95_Rel_DLR2_p;    CI95_Vio_DLR2_p;    CI95_Rel_CRL_p], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_p);
writetable(Summary_p, 'Summary_Polynomial.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_r   = mean(Rel_DLR_r,  2, 'omitnan');
Viol_DLR_mean_by_region_r  = mean(Viol_DLR_r, 2, 'omitnan');
Rel_DLR2_mean_by_region_r  = mean(Rel_DLR2_r,  2, 'omitnan');
Viol_DLR2_mean_by_region_r = mean(Viol_DLR2_r, 2, 'omitnan');
Rel_CRL_mean_by_region_r   = mean(Rel_CRL_r, 2, 'omitnan');

Mean_Rel_DLR_r  = mean(Rel_DLR_mean_by_region_r,  'omitnan');   CI95_Rel_DLR_r  = 1.96 * std(Rel_DLR_mean_by_region_r,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_r));
Mean_Vio_DLR_r  = mean(Viol_DLR_mean_by_region_r, 'omitnan');   CI95_Vio_DLR_r  = 1.96 * std(Viol_DLR_mean_by_region_r, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_r));
Mean_Rel_DLR2_r = mean(Rel_DLR2_mean_by_region_r, 'omitnan');   CI95_Rel_DLR2_r = 1.96 * std(Rel_DLR2_mean_by_region_r, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_r));
Mean_Vio_DLR2_r = mean(Viol_DLR2_mean_by_region_r,'omitnan');   CI95_Vio_DLR2_r = 1.96 * std(Viol_DLR2_mean_by_region_r,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_r));
Mean_Rel_CRL_r  = mean(Rel_CRL_mean_by_region_r,  'omitnan');   CI95_Rel_CRL_r  = 1.96 * std(Rel_CRL_mean_by_region_r,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_r));

Summary_r = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_r;  Mean_Vio_DLR_r;  Mean_Rel_DLR2_r;  Mean_Vio_DLR2_r;  Mean_Rel_CRL_r], ...
    [CI95_Rel_DLR_r;    CI95_Vio_DLR_r;    CI95_Rel_DLR2_r;    CI95_Vio_DLR2_r;    CI95_Rel_CRL_r], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_r);
writetable(Summary_r, 'Summary_RBF.csv');
% ---------------------------------------------------------------------------------------------------

Rel_DLR_mean_by_region_s   = mean(Rel_DLR_s,  2, 'omitnan');
Viol_DLR_mean_by_region_s  = mean(Viol_DLR_s, 2, 'omitnan');
Rel_DLR2_mean_by_region_s  = mean(Rel_DLR2_s,  2, 'omitnan');
Viol_DLR2_mean_by_region_s = mean(Viol_DLR2_s, 2, 'omitnan');
Rel_CRL_mean_by_region_s   = mean(Rel_CRL_s, 2, 'omitnan');

Mean_Rel_DLR_s  = mean(Rel_DLR_mean_by_region_s,  'omitnan');   CI95_Rel_DLR_s  = 1.96 * std(Rel_DLR_mean_by_region_s,  0, 'omitnan') / sqrt(numel(Rel_DLR_mean_by_region_s));
Mean_Vio_DLR_s  = mean(Viol_DLR_mean_by_region_s, 'omitnan');   CI95_Vio_DLR_s  = 1.96 * std(Viol_DLR_mean_by_region_s, 0, 'omitnan') / sqrt(numel(Viol_DLR_mean_by_region_s));
Mean_Rel_DLR2_s = mean(Rel_DLR2_mean_by_region_s, 'omitnan');   CI95_Rel_DLR2_s = 1.96 * std(Rel_DLR2_mean_by_region_s, 0, 'omitnan') / sqrt(numel(Rel_DLR2_mean_by_region_s));
Mean_Vio_DLR2_s = mean(Viol_DLR2_mean_by_region_s,'omitnan');   CI95_Vio_DLR2_s = 1.96 * std(Viol_DLR2_mean_by_region_s,0, 'omitnan') / sqrt(numel(Viol_DLR2_mean_by_region_s));
Mean_Rel_CRL_s  = mean(Rel_CRL_mean_by_region_s,  'omitnan');   CI95_Rel_CRL_s  = 1.96 * std(Rel_CRL_mean_by_region_s,  0, 'omitnan') / sqrt(numel(Rel_CRL_mean_by_region_s));

Summary_s = table( ...
    ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
    [Mean_Rel_DLR_s;  Mean_Vio_DLR_s;  Mean_Rel_DLR2_s;  Mean_Vio_DLR2_s;  Mean_Rel_CRL_s], ...
    [CI95_Rel_DLR_s;    CI95_Vio_DLR_s;    CI95_Rel_DLR2_s;    CI95_Vio_DLR2_s;    CI95_Rel_CRL_s], ...
    'VariableNames', {'Metric','Mean','StdDev'} ...
);
disp(Summary_s);
writetable(Summary_s, 'Summary_SVD.csv');





% Mean_Rel_DLR_s_  = mean(Rel_DLR_s(:),  'omitnan');   CI95_Rel_DLR_s_  = std(Rel_DLR_s(:),  0, 'omitnan');
% Mean_Vio_DLR_s_  = mean(Viol_DLR_s(:), 'omitnan');   CI95_Vio_DLR_s_  = std(Viol_DLR_s(:), 0, 'omitnan');
% Mean_Rel_DLR2_s_ = mean(Rel_DLR2_s(:), 'omitnan');   CI95_Rel_DLR2_s_ = std(Rel_DLR2_s(:), 0, 'omitnan');
% Mean_Vio_DLR2_s_ = mean(Viol_DLR2_s(:),'omitnan');   CI95_Vio_DLR2_s_ = std(Viol_DLR2_s(:),0, 'omitnan');
% Mean_Rel_CRL_s_  = mean(Rel_CRL_s(:),  'omitnan');   CI95_Rel_CRL_s_  = std(Rel_CRL_s(:),  0, 'omitnan');
% 
% Summary_s_ = table( ...
%     ["Rel_DLR"; "Viol_DLR"; "Rel_DLR2"; "Viol_DLR2"; "Rel_CRL"], ...
%     [Mean_Rel_DLR_s_;  Mean_Vio_DLR_s_;  Mean_Rel_DLR2_s_;  Mean_Vio_DLR2_s_;  Mean_Rel_CRL_s_], ...
%     [CI95_Rel_DLR_s_;    CI95_Vio_DLR_s_;    CI95_Rel_DLR2_s_;    CI95_Vio_DLR2_s_;    CI95_Rel_CRL_s_], ...
%     'VariableNames', {'Metric','Mean','StdDev'} ...
% );
% disp(Summary_s_);
% writetable(Summary_s_, 'Summary_SVD_.csv');