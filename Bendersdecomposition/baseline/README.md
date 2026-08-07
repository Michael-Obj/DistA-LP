# Baseline Artifact Guide

This directory contains the baseline implementations used for the experiments in the DISTA-LP paper. The primary runners aggregate the experimental results in memory, calculate the mean and standard deviation, and print paper-style tables directly in the MATLAB Command Window.

The primary runners do **not** save newly generated result files.

## 1. Quick start

Start MATLAB and change the working directory to the root of this folder:

```matlab
cd('<path-to-artifact>/baseline')
```

Then run one of the following entry files:

```matlab
run('EM_EMBR.m')
run('laplace_baseline.m')
run('PAnDA_baseline.m')
```

All paths used by these scripts are relative to the baseline root directory. No machine-specific absolute path needs to be changed.

## 2. Baseline entry files

| Baseline | Entry file | Main implementation |
|---|---|---|
| Exponential Mechanism (EM) | `EM_EMBR.m` | `functions/loss_for_benchmark.m` |
| Bayesian Remapping (RMP) | `EM_EMBR.m` | `functions/loss_for_benchmark.m` |
| Coarse-grid LP approximation (LP-A) | `EM_EMBR.m` | `coarse.m` |
| Planar Laplace | `laplace_baseline.m` | `planar_laplace_utility_loss.m` |
| PAnDA | `PAnDA_baseline.m` | `PAnDA.m` |

`EM_EMBR.m` runs EM, RMP, and LP-A together. The other two entry files run Planar Laplace and PAnDA, respectively.

## 3. Experiment configuration

Each primary entry file contains a configuration block near the top:

```matlab
city = 'london';              % 'rome', 'london', or 'nyc'
node_count = 2000;            % 2000, 4000, or 6000
epsilons = [4, 7, 10];
user_ids = 1:10;
```

To run another supplied experiment, change only `city` and `node_count`. The script automatically constructs the corresponding input-data, road-network, and graph paths.

Supported settings are:

| Setting | Values |
|---|---|
| City | `rome`, `london`, `nyc` |
| Number of records | `2000`, `4000`, `6000` |
| Privacy budget | `4`, `7`, `10` km^-1 |
| Users | `1:10` |

The runners automatically select the available repetitions:

| Dataset | Repetitions |
|---|---:|
| London, all sizes | 6 |
| NYC, all sizes | 4 |
| Rome, 2,000 records | 5 |
| Rome, 4,000 or 6,000 records | 4 |

Shared default parameters are defined in:

```text
functions/parameters.m
```

Important fields include:

```matlab
env_parameters.NEIGHBOR_THRESHOLD
env_parameters.NR_AGENT
env_parameters.NR_OBFLOC
env_parameters.EPSILON
env_parameters.NR_NODE_IN_TARGET
env_parameters.ITER
```

The primary runners set `EPSILON`, `NR_NODE_IN_TARGET`, and `NR_OBFLOC` when needed, so these fields normally do not require manual changes.

## 4. Statistical aggregation

The scripts use the following aggregation procedure:

1. For every repetition, utility loss is summed across the ten users.
2. For every repetition, runtime and violation ratio are averaged across the ten users.
3. The mean and sample standard deviation are calculated across repetitions.
4. Utility loss is divided by 10,000 before printing, matching the scale used in the paper table.

The current output is reported as:

```text
mean +/- standard deviation
```

This is intentionally different from a `mean +/- 1.96 x standard error` confidence interval. If the paper table must be reproduced using the latter convention, replace the printed deviation with:

```matlab
1.96 * standard_deviation / sqrt(number_of_repetitions)
```

## 5. Printed output

Each runner prints three tables where applicable:

1. Utility loss
2. Violation ratio
3. Computation time

Example:

```text
Baseline setting: city=LONDON, records=2000, users=10, repetitions=6

Utility loss (10,000 meters) -- mean +/- standard deviation
Method       | epsilon=4          | epsilon=7          | epsilon=10
-------------------------------------------------------------------
EM           | mean +/- std       | mean +/- std       | mean +/- std
RMP          | mean +/- std       | mean +/- std       | mean +/- std
LP-A         | mean +/- std       | mean +/- std       | mean +/- std
```

No result MAT or CSV file is created by the three primary runners. The final values are displayed in the MATLAB Command Window.

## 6. EM, RMP, and LP-A

Run:

```matlab
run('EM_EMBR.m')
```

The runner evaluates epsilon values 4, 7, and 10 in one execution.

### EM

EM constructs a discrete exponential mechanism using:

```matlab
exp(-epsilon * distance / 2)
```

The reported values include utility loss and runtime. Its violation ratio is printed as zero because EM satisfies the target privacy definition by construction.

### RMP

RMP applies Bayesian remapping to the EM mechanism. The reported values include remapped utility loss and remapping runtime. Its violation ratio is also printed as zero because remapping is privacy-preserving post-processing.

### LP-A

LP-A is the coarse-grid LP approximation implemented in `coarse.m`. The code uses an 8-by-8 geographic grid and solves the resulting problem with `linprog`.

LP-A reports:

- Utility loss
- Runtime
- Empirical distance-approximation violation ratio

LP-A is substantially slower than EM and RMP. A complete run executes one LP for every user, repetition, and privacy budget, so large settings may take a considerable amount of time.

## 7. Planar Laplace

Run:

```matlab
run('laplace_baseline.m')
```

The mechanism is implemented in:

```text
planar_laplace_utility_loss.m
```

The runner evaluates epsilon values 4, 7, and 10 in one execution and prints utility loss, runtime, and violation ratio.

The Planar Laplace violation ratio is printed as zero because the mechanism satisfies the target privacy definition by construction. The runner does not execute the previous exhaustive three-level violation-checking loop.

## 8. PAnDA

Run:

```matlab
run('PAnDA_baseline.m')
```

The standard implementation is:

```text
PAnDA.m
```

`PAnDA_baseline.m` deliberately calls `PAnDA.m`, not `PAnDA_memory_instrumented.m`, so the primary PAnDA run does not create memory checkpoint files.

PAnDA evaluates epsilon values 4, 7, and 10 in one execution. Its printed tables contain two method rows:

| Row | Meaning |
|---|---|
| `PAnDA` | Main PAnDA mechanism |
| `LB` | Companion lower-bound solve included in the implementation |

The main parameters in `PAnDA.m` are:

```matlab
lambda = 0.5;
alpha_hat = 0.95;
delta = 1e-7;
range_threshold = D_MAX / 150;
env_parameters.NR_AGENT = 25;
ITER = 100;
```

PAnDA may also require considerable time for large datasets because it repeatedly solves decomposed optimization problems.

## 9. Input-data layout

Sampled experiment inputs follow this structure:

```text
<city>_location_data_<number_of_records>_nodes/
    location_data_sample_<user>/
        location_data_r<repeat>_user<user>.mat
```

Each MAT file contains variables such as:

- `node_tar`: selected true locations
- `obf_ID`: candidate perturbed-output locations
- `lon_sel`: selected longitudes, where available
- `lat_sel`: selected latitudes, where available

Raw road-network data are stored under:

```text
Dataset/<city>/raw/<city>_nodes.csv
Dataset/<city>/raw/<city>_edges.csv
```

Precomputed MATLAB graph objects are stored in the baseline root:

```text
G_<city>.mat
u_<city>.mat
v_<city>.mat
```

## 10. Software requirements

The code was checked with MATLAB R2025b. A recent MATLAB version should be sufficient.

Required toolboxes:

- Optimization Toolbox (`linprog`)
- Statistics and Machine Learning Toolbox (`kmeans`, `pdist2`)
- Symbolic Math Toolbox (`heaviside`)

The artifact includes the Haversine helper package under:

```text
functions/haversine/
```

## 11. Full LP and COPT

A runnable COPT implementation is not included in this baseline directory.

The full centralized LP is computationally infeasible at the evaluated domain sizes and is reported as unavailable or exceeding the 1,800-second timeout where applicable.

`main_BD.m` is legacy/development code and is not an entry file for the baseline results described in this guide.

## 12. Auxiliary and precomputed files

The directory contains previously generated MAT files under folders such as:

```text
baseline_laplace/
baseline_PAnDA/
baseline_<city>_<number_of_records>/
```

These are precomputed historical results. The current primary runners do not read them when calculating new results and do not overwrite them.

Files whose names contain `memory`, including `PAnDA_memory_instrumented.m` and scripts under `memory_cost_probe_outputs/`, are auxiliary profiling utilities rather than primary baseline runners. Some of those auxiliary scripts may write profiling checkpoints if run directly.

## 13. Recommended reproduction procedure

For each city and record count:

1. Set `city` and `node_count` at the top of the selected runner.
2. Run the script from the baseline root directory.
3. Wait for all users, repetitions, and epsilon values to complete.
4. Copy the three printed tables from the MATLAB Command Window.
5. Repeat for the remaining city/size combinations.

For a quick environment check, reduce `user_ids` to one user before launching the complete experiment:

```matlab
user_ids = 1;
```

Restore `user_ids = 1:10` for the reported artifact results.
