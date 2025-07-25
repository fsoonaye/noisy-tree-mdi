# Analysis of Mean Decrease in Impurity (MDI) by Tree Depth

This repository contains the code for investigating the behavior of the Mean Decrease in Impurity (MDI) metric at different depths within decision trees and random forests for regression. The primary goal is to analyze how feature importance evolves when injecting additional noise features.

The official `DecisionTree.jl` package has been forked to enable the extraction of MDI scores **at each depth of the tree**. This modification allows for efficient Monte Carlo simulations by growing N trees at **max_depth** rather than building N trees at every depth from 1 to **max_depth**, significantly reducing computation time. The modified library can be found [here](https://github.com/fsoonaye/DecisionTree.jl).

## Prerequisites

To run the code in this repository, you will need to have Julia installed on your system. You can download and install the latest version of Julia from the [official website](https://julialang.org/downloads/).

Follow these steps to set up the project locally and install the required dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fsoonaye/noisy-tree-mdi.git
   cd noisy-tree-mdi
   ```

2. **Install Julia dependencies:**
   Start a Julia REPL in the project directory and run the following commands to install all the required packages.
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

## Usage

The project is divided into two main scripts: one for running simulations and one for generating plots. Simulation results are stored in portable Arrow and JSON formats, enabling easy analysis and visualization with other programming languages. You can examine how data is stored in `sim.jl` and how it's loaded and processed in `data.jl` to get inspiration for analyzing and visualizing the generated data using other tools, such as Python with the matplotlib library.

### 1. Running a Simulation

The `run_sim.jl` script executes a simulation based on the hyperparameters defined within the file. The simulations are optimized for multi-threading. Use the `--threads` flag to specify the number of threads.

```bash
julia --threads auto run_sim.jl
```

### 2. Generating Plots

The `view_sim.jl` script processes the raw data from a simulation and generates the corresponding plots. You must provide the name of the experiment directory as a command-line argument.

```bash
julia view_sim.jl d3_p2_m2_k15_n200000_M1000
```
