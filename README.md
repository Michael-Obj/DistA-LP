# DISTA-LP: Privacy-Preserving Distributed Linear Programming for Metric Differential Privacy

**Paper Title:** *DISTA-LP: Privacy-Preserving Distributed Linear Programming for Metric Differential Privacy*

## Description

This repository contains the source code and experimental framework for **DISTA-LP**, a privacy-preserving distributed optimization framework that enables scalable metric differential privacy (mDP) enforcement on large-scale, fine-grained domains such as road networks.

### Key Innovation

DISTA-LP addresses a critical challenge in distributed privacy-preserving systems: uploading user-specific LP coefficient matrices (distance and loss blocks) can itself leak sensitive information about underlying records. The framework mitigates this risk through:

1. **Coefficient Surrogatization**: Replaces high-dimensional coefficient matrices with compact low-dimensional surrogates (parametric functions or low-rank SVD approximations)
2. **Private Parameter Release**: Perturbs surrogate parameters with calibrated, asymmetric noise before upload
3. **Probabilistic mDP Accounting**: Uses a sampling-based auditor to empirically quantify privacy loss from parameter release
4. **Benders Decomposition**: Decomposes the global LP into scalable master/subproblem formulations for efficient distributed solving

**Main Result**: DISTA-LP achieves near-zero mDP violation rates while reducing computation time substantially compared to monolithic and hybrid baselines, with improved or competitive utility across large real-world road-network datasets.

### Directory Structure

```
DistributedLP/
├── README.md                          # This file
├── parameters.m                       # Global parameter configuration
├── main.m                             # Example usage script
├── main_test_*.m                      # Test scripts for different surrogate families
│   ├── main_test_Gaussian.m           # Tests 2D Gaussian parametric surrogates
│   ├── main_test_Polynomial.m         # Tests 2D polynomial parametric surrogates
│   ├── main_test_RBF.m                # Tests RBF parametric surrogates
│   └── main_test_Lowrank_SVD.m        # Tests low-rank SVD surrogates
├── Bendersdecomposition/              # Main DISTA-LP framework implementation
│   ├── main.m                         # Primary experimental runner
│   ├── parameters.m                   # Framework-specific parameters
│   ├── PL_mean_std.m                  # Privacy loss statistics computation
│   ├── loss_for_benchmark.m           # Benchmark loss evaluation
│   ├── coarse.m                       # Coarse-grid baseline implementation
│   ├── classes/                       # Object-oriented components
│   │   ├── Server/                    # Server-side LP solver
│   │   ├── User/                      # User-side surrogate fitting
│   │   ├── MasterProgram/             # Benders master problem
│   │   └── Subproblem/                # Benders subproblems
│   ├── func/                          # Core algorithmic functions
│   │   ├── benchmarks/                # Baseline methods (EM, Laplace, TEM, LP, COPT, etc.)
│   │   ├── read_files/                # Dataset I/O utilities
│   │   └── haversine/                 # Haversine distance calculations
│   ├── dataset/                       # Dataset configuration
│   ├── rome_location_data_*/          # Preprocessed Rome location subsets (2K-10K nodes)
│   ├── london_location_data_*/        # Preprocessed London location subsets
│   ├── nyc_location_data_*/           # Preprocessed NYC location subsets
│   ├── rome_baseline_results/         # Rome experimental results
│   ├── london_baseline_results/       # London experimental results
│   ├── nyc_baseline_results/          # NYC experimental results
│   └── results/                       # Output directory for latest experiments
├── functions/                         # Legacy/utility functions
├── figure_tools/                      # Visualization helpers
├── datasets/                          # Additional dataset resources
└── PL_Experiment_Results/             # Privacy loss experiment outputs
```

## System Requirements

### Recommended Hardware
- **Processor**: Intel Core i9 or equivalent (24+ cores recommended for parallel subproblem solving)
- **Memory**: 32 GB DDR5 RAM (tested on 32 GB; 64+ GB for larger datasets)
- **Disk Space**: 5-10 GB for datasets, code, and experimental results
- **GPU** (optional): NVIDIA GeForce RTX 4090 or similar (for potential acceleration)

### Supported Operating Systems
- **Windows 10/11**
- **macOS Monterey/Ventura/Sonoma**
- **Ubuntu Linux 20.04/22.04**

## Environment Setup

### MATLAB Requirements
- **MATLAB R2024b** or later
- **Required Toolboxes**:
  - **Optimization Toolbox** (for `linprog` and Benders decomposition)
  - **Statistics and Machine Learning Toolbox** (for `randsample` and statistical functions)
  - **Parallel Computing Toolbox** (optional, for distributed subproblem solving)

### Installation
1. Install MATLAB with the above toolboxes
2. Clone or download this repository
3. In MATLAB, navigate to the repository root directory
4. The code will automatically add required paths via `addpath()` commands in main scripts

## Quick Start

### Running the DISTA-LP Framework

#### Basic Experiment (Benders Decomposition Framework)
```matlab
% Navigate to Bendersdecomposition/ directory
cd Bendersdecomposition

% Run the main experimental script
main

% This will:
% 1. Load road-network datasets (Rome, London, NYC)
% 2. Execute the DISTA-LP pipeline with Benders decomposition
% 3. Test multiple surrogate approximation methods
% 4. Generate utility loss, violation ratios, and runtime metrics
% 5. Save results to ./results/ subdirectory
```

#### Testing Individual Surrogate Families
From the root directory, run any of the test scripts to evaluate specific surrogate approximation methods:

```matlab
% Test 2D Gaussian parametric surrogates
main_test_Gaussian.m

% Test 2D polynomial parametric surrogates
main_test_Polynomial.m

% Test RBF parametric surrogates
main_test_RBF.m

% Test low-rank SVD surrogates (recommended for best privacy-accuracy tradeoff)
main_test_Lowrank_SVD.m
```

#### Configuring Experiments
Edit `Bendersdecomposition/parameters.m` to adjust:

```matlab
env_parameters.EPSILON = 10.0;        % Privacy budget (ε)
env_parameters.GAMMA = 20.0;          % Relevant location distance threshold (km)
env_parameters.NR_TASK = 50;          % Number of tasks
env_parameters.deg = 3;               % Polynomial degree
env_parameters.NUM_CENTRES = 25;      % RBF center count
env_parameters.rank_r = 5;            % Low-rank SVD rank
```

#### Target Region Selection
Modify longitude/latitude bounds in main scripts:

```matlab
% Example: Rome subregion (2,000 nodes)
TARGET_LON_MAX = 12.4; 
TARGET_LON_MIN = 12.2; 
TARGET_LAT_MAX = 42.1;
TARGET_LAT_MIN = 41.901;
```

## Core Features

### 1. Surrogate-Based Coefficient Approximation

**Four Surrogate Families** (ranked by fidelity-privacy tradeoff):

| Surrogate | Parameters | Accuracy | Privacy | Use Case |
|-----------|-----------|----------|---------|----------|
| **Low-Rank SVD** | 30–50 | Very High | ★★★★★ | Recommended (best balance) |
| **2D Gaussian** | 6 | Low | ★★★★ | Smooth, bell-shaped data |
| **2D Polynomial** | 10 | Medium | ★★★ | Global trends |
| **RBF** | 25 | High | ★★ | Multi-peak irregular data |

**Key Innovation**: Replaces high-dimensional coefficient matrices with low-dimensional surrogates, then perturbs only the compact parameters, drastically reducing privacy leakage from coefficient uploads.

### 2. Private Parameter Release

Three calibrated perturbation mechanisms:

- **M^p_dx (intra-neighborhood distance)**: Asymmetric Laplace noise that biases downward to tighten constraints
- **M^p_dy (cross-domain distance)**: Calibrated perturbation linking neighborhoods to shared output space
- **M^p_cy (utility-loss parameters)**: Protects sensitive utility-loss structure via biased upward noise

**Structure-Preserving Design**: Perturbation respects metric properties (non-negativity, symmetry, triangle inequalities) via projection and post-processing.

### 3. Probabilistic mDP Accounting

- **Probabilistic mDP (PmDP)**: Empirically quantifies privacy loss of continuous parameter releases via Monte Carlo sampling
- **Posterior Leakage Auditor**: Estimates privacy-loss distributions to allocate budgets across four upload stages
- **Sequential Composition**: Ensures end-to-end privacy bound: epsilon_dx + epsilon_dy + epsilon_cy + epsilon_pt ≤ epsilon

### 4. Benders Decomposition

**Two-Stage Optimization**:
- **Master Problem**: Optimizes global WEM weights and aggregate utility loss
- **Subproblems**: Each user solves independent neighborhood-scale LPs, generating cutting planes to refine master
- **Benefit**: Scales from 500 (centralized LP) to 15,000+ nodes per region

### 5. Weighted Exponential Mechanism Integration

**Hybrid Approach**: Combines LP optimization (utility-critical near-outputs) with pre-defined exponential-decay structure (distant, low-impact regions) to reduce problem dimensionality.

## Main Experimental Results

All results reported with 95% confidence intervals. Full details in the paper.

### Result 1: mDP Violation Ratios

**Finding**: DISTA-LP achieves **0.00 ± 0.00** violation ratios across all datasets and privacy budgets, matching predefined-noise baselines (Laplace, EM) while outperforming centralized LP (50–59% violations).

**Key Datasets**: Rome, London, NYC road networks (2,000–10,000 nodes)  
**Privacy Budgets**: epsilon in {4.0, 7.0, 10.0} km^-1

### Result 2: Utility Loss

**Finding**: DISTA-LP reduces expected utility loss by ~50% vs. predefined mechanisms and ~49% vs. PAnDA hybrid method, while remaining within 1.5–3× of the universal lower bound.

**Example (Rome, 4,000 nodes, epsilon = 4.0 km^-1)**:
- DISTA-LP: **2.88 ± 0.21** (10,000 m units)
- EM/Laplace: 7.16 ± 2.53
- LP: 6.76 ± 2.55
- Lower Bound: ~1.7

### Result 3: Computational Efficiency

**Finding**: DISTA-LP achieves favorable runtime scaling vs. centralized approaches:
- Centralized LP/COPT: ≥1,800 seconds (timeout)
- DISTA-LP: 2–5 seconds
- Hybrid (PAnDA): 0.1–0.3 seconds (lower utility)

**Scalability**: Stable performance across dataset sizes (2K–10K nodes); centralized methods exhibit steep growth.

## Output and Results

### Result Directories
```
Bendersdecomposition/results/
├── rome_baseline_results/          # Rome experiments
├── london_baseline_results/        # London experiments  
├── nyc_baseline_results/           # NYC experiments
│
└── Each contains:
    ├── Summary_Gaussian.csv        # Gaussian surrogate metrics
    ├── Summary_Polynomial.csv      # Polynomial surrogate metrics
    ├── Summary_RBF.csv             # RBF surrogate metrics
    ├── Summary_SVD.csv             # Low-rank SVD metrics (recommended)
    └── *.mat files                 # Raw computation time, violations, costs
```

### Output Metrics

Each experiment produces:
- **Utility Loss**: Expected distance between true and perturbed locations
- **mDP Violation Ratio**: Fraction of tested constraints exceeding target privacy budget
- **Relative Error (RelErr)**: Approximation fidelity vs. original coefficients
- **Privacy Loss (PL)**: Empirical privacy loss distribution from uploads
- **Computation Time**: Total runtime in seconds

### CSV Format Example
```
dataset,n_nodes,epsilon,method,utility_loss,violation_ratio,privacy_loss,runtime_sec
Rome,2000,4.0,SVD,3.47±0.24,0.00±0.00,1.25±0.23,2.03±0.16
Rome,2000,4.0,Polynomial,5.19±0.91,0.59±0.04,6.42±4.64,11.27±0.12
```

## Code Organization

### Key Classes (Object-Oriented Design)

**Server** (classes/Server/)
- Aggregates uploaded surrogate parameters
- Reconstructs approximate coefficient matrices
- Solves master problem via Benders decomposition

**User** (classes/User/)
- Constructs local coefficient blocks
- Fits surrogates and perturbs parameters
- Handles local permutations for privacy

**MasterProgram** (classes/MasterProgram/)
- Formulates and solves WEM+LP master problem
- Manages cut generation from subproblems
- Converges to near-optimal solution

**Subproblem** (classes/Subproblem/)
- Solves independent neighborhood-scale LPs
- Generates dual cuts for feasibility/optimality
- Validates guessed multiplier values

### Core Functions (func/)

| Function | Purpose |
|----------|---------|
| construct_distance_matrix() | Compute pairwise Haversine distances |
| fit_surrogate_gaussian() | Fit 2D Gaussian parametric model |
| fit_surrogate_polynomial() | Fit polynomial basis functions |
| fit_surrogate_rbf() | Fit RBF network |
| fit_surrogate_svd() | Compute truncated SVD approximation |
| perturb_parameters() | Apply calibrated Laplace noise to parameters |
| audit_privacy_loss() | Estimate privacy loss via Monte Carlo sampling |
| reconstruct_coefficients() | Recover approximate matrices from perturbed parameters |
| solve_benders_master() | Solve WEM+LP master problem |
| solve_benders_subproblem() | Solve neighborhood subproblem |

## Datasets

### Road Networks (OpenStreetMap)

All datasets preprocessed from OpenStreetMap (OSM) graph data.

| City | Nodes | Edges | Coverage |
|------|-------|-------|----------|
| **Rome, Italy** | 43,160 | 89,739 | Central urban area |
| **London, UK** | 12,820 | 299,524 | Central districts (Westminster, City, boroughs) |
| **NYC, USA** | 55,353 | 139,638 | Lower & mid-Manhattan, adjacent neighborhoods |

### Subregion Data

For each city, axis-aligned rectangular subregions (2,000–10,000 nodes) are provided:

```
london_location_data_2000_nodes/      % ~2K nodes
london_location_data_4000_nodes/      % ~4K nodes
london_location_data_6000_nodes/      % ~6K nodes
london_location_data_8000_nodes/      % ~8K nodes
london_location_data_10000_nodes/     % ~10K nodes
```

**Format**: CSV files with OSM node IDs, coordinates (longitude, latitude), graph adjacency

### Loading Datasets

```matlab
opts = detectImportOptions('./datasets/rome/rome_nodes.csv');
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable('./datasets/rome/rome_nodes.csv', opts);
df_edges = readtable('./datasets/rome/rome_edges.csv');

lon = table2array(df_nodes(:, 'x'));
lat = table2array(df_nodes(:, 'y'));
```

## Troubleshooting & Notes

### Common Issues

1. **"Function not found" error**  
   - Ensure all addpath() commands have executed
   - Check parameters.m is in the active directory

2. **Out of memory (OOM) on large datasets**  
   - Reduce NR_TASK or NUM_CENTRES in parameters.m
   - Use smaller subregions (e.g., 2K–4K nodes)
   - Enable parallel computing with parpool for subproblems

3. **Linprog timeout (1,800 sec)**  
   - This is expected for centralized LP on 4K+ nodes
   - DISTA-LP should complete in 2–5 seconds
   - If not, check Benders convergence criteria in master/subproblem solvers

4. **High privacy loss from uploads**  
   - Reduce noise scale b in perturbation mechanisms
   - Use low-rank SVD (fewer parameters than parametric surrogates)
   - Increase rank r if RelErr is too high


## Related Work & Baselines

### Implemented Baselines

- **Laplace Mechanism**: Additive Laplace noise calibrated to metric distance
- **Exponential Mechanism (EM)**: Probabilistic output selection via utility-driven weights
- **TEM (Truncated EM)**: Variant with clipping for bounded support
- **LP**: Centralized linear programming (baseline, poor scalability)
- **COPT**: Hybrid LP+EM on utility-critical subset 
- **RMP (Bayesian Remapping)**: Post-processing utility improvement 
- **PAnDA**: Anchor-based distributed LP
- **Coarse Grid LP**: Discretized LP on coarse spatial grid
