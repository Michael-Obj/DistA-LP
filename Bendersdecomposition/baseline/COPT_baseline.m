%% COPT baseline runner
% Run from the baseline artifact root. Full paper-scale cases are
% preflighted before their very large LP matrices are allocated.

clear;
clc;
rng(1);

%% Experiment configuration
city = 'rome';                 % 'rome', 'london', or 'nyc'
node_count = 2000;            % 2000, 4000, or 6000
epsilons = [4, 7, 10];
user_ids = 1:10;

% COPT configuration from the supplied implementation.
copt_lambda = 10;
copt_neighbors = 2000;

% Full-scale safety and timeout settings.
max_time_seconds = 1800;
max_setup_bytes = 16 * 1024^3;
allow_oversized = false;

% Leave empty for the supplied paper-scale domain. Set to 20 or more for a
% small interface smoke test; all 20 supplied output locations are kept.
record_limit = [];

addpath('./functions');
addpath('./functions/haversine');
addpath('./COPT');

config = struct('city', city, 'node_count', node_count, ...
    'epsilons', epsilons, 'user_ids', user_ids, ...
    'record_limit', record_limit, 'max_time_seconds', max_time_seconds, ...
    'max_setup_bytes', max_setup_bytes, ...
    'allow_oversized', allow_oversized, ...
    'copt_lambda', copt_lambda, 'copt_neighbors', copt_neighbors);

copt_results = run_optimization_baseline('COPT', config);
