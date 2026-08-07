addpath('./classes/Server/');
addpath('./classes/User/');
addpath('./classes/MasterProgram/');
addpath('./classes/Subproblem/');
addpath('./func/benchmarks/');
addpath('./func/benchmarks/randl/');
addpath('./func'); 
addpath('./func/read_files'); 
addpath('./func/haversine');

parameters;
baseline_fn = @() loss_for_benchmark;   % Baseline call
dista_fn    = @() main;                 % Benders implementation
results = memory_cost_eval(env_parameters, baseline_fn, dista_fn);


function results = memory_cost_eval(env_parameters, lp_solver_handle, dista_solver_handle)
    % addpath(fullfile(pwd, 'Bendersdecomposition', 'classes'));

    results = struct();

    % Helper to measure memory and runtime for a given solver
    function [runtime, mem_used] = run_and_measure(solver_func)
        % Reset MATLAB memory counter
        [mem_before,~] = memory;
        t0 = tic;
        try
            solver_func();
        catch ME
            warning('Solver execution failed: %s', ME.message);
        end
        runtime = toc(t0);
        [mem_after,~] = memory;
        mem_used = mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB;
    end

    % Measure baseline solver
    if ~isempty(lp_solver_handle)
        [runtime_lp, mem_lp] = run_and_measure(lp_solver_handle);
        results.lp_runtime_sec = runtime_lp;
        results.lp_peak_mem_bytes = mem_lp;
    else
        results.lp_runtime_sec = NaN;
        results.lp_peak_mem_bytes = NaN;
    end

    % Measure DISTA‑LP solver
    if ~isempty(dista_solver_handle)
        [runtime_dista, mem_dista] = run_and_measure(dista_solver_handle);
        results.dista_runtime_sec = runtime_dista;
        results.dista_peak_mem_bytes = mem_dista;
    else
        results.dista_runtime_sec = NaN;
        results.dista_peak_mem_bytes = NaN;
    end
    disp(results);
end