using DecisionTree
using Random
using Printf
using Statistics
using Distributions
using DataFrames
using Arrow
using JSON
using ProgressMeter
using Parameters

include("types.jl")

"""
Generate synthetic dataset with signal and noise features based on a linear model.
"""
function gen_linear_data(
    n_samples::Int,
    signal_coeffs::Vector{Float64},
    n_feat_noise::Int,
    noise_std::Float64,
    rng::Random.AbstractRNG,
)
    n_feat_signal = length(signal_coeffs)

    # Generate signal features (uniform distribution)
    x_signal = rand(rng, n_samples, n_feat_signal)

    # Generate noise features (normal distribution)
    z_noise = randn(rng, n_samples, n_feat_noise)

    # Combine into a single feature matrix
    x_noisy = hcat(x_signal, z_noise)

    # Generate the target variable (y = X*β + ε)
    noise = randn(rng, n_samples) .* noise_std
    y = x_signal * signal_coeffs .+ noise

    return x_signal, x_noisy, y
end

"""
Generate a systematic filename from hyperparameters struct.
"""
function pms_to_exp_name(pms::Hyperparameters)
    parts = [
        "d$(pms.N_FEAT_SIGNAL)",
        "p$(pms.N_FEAT_NOISY)",
        "m$(pms.MTRY)",
        "k$(pms.MAX_DEPTH)",
        "n$(pms.N_SAMPLES)",
        "M$(pms.N_TREES)",
    ]
    return join(parts, "_")
end


function run_sim(pms::Hyperparameters; results_dir="results")
    # 1. Setup and print parameters
    println("Initializing Simulation...")

    # Check for existing results directory before running the simulation
    exp_name = pms_to_exp_name(pms)
    exp_dir = joinpath(results_dir, exp_name)
    if isdir(exp_dir)
        error(
            "Results directory already exists for this parameter set: `$exp_dir`\n" *
            "Please move or delete it before re-running to avoid overwriting results.",
        )
    end

    println("Parameters:")
    for field in fieldnames(Hyperparameters)
        println("    - ", rpad(string(field), 15), ": ", getfield(pms, field))
    end
    println("    - THREADS        : ", Threads.nthreads())
    println("\n")

    N_FEAT_TOTAL = pms.N_FEAT_SIGNAL + pms.N_FEAT_NOISY
    master_rng = MersenneTwister(pms.RNG_SEED)
    all_mdi_noisy = zeros(N_FEAT_TOTAL, pms.MAX_DEPTH, pms.N_TREES)
    all_mdi_signal = zeros(pms.N_FEAT_SIGNAL, pms.MAX_DEPTH, pms.N_TREES)

    # Generate synthetic data once for all runs
    X_signal, X_noisy, y = gen_linear_data(
        pms.N_SAMPLES, pms.SIGNAL_COEFFS, pms.N_FEAT_NOISY, pms.EPS_STD, master_rng
    )

    # 2. Fitting all trees
    p = Progress(pms.N_TREES, "Fitting $(pms.N_TREES) noisy and signal trees:")

    master_seed = rand(master_rng, UInt)
    @time Threads.@threads for i in 1:pms.N_TREES
        # Create a thread-safe RNG for each iteration
        local_rng = MersenneTwister(master_seed + i)

        # Fit noisy tree
        noisy_tree = build_tree(
            y,
            X_noisy,
            pms.MTRY,
            pms.MAX_DEPTH,
            pms.NODESIZE;
            rng=local_rng,
            impurity_importance=true,
        )

        # Fit signal tree
        signal_tree = build_tree(
            y,
            X_signal,
            pms.MTRY,
            pms.MAX_DEPTH,
            pms.NODESIZE;
            rng=local_rng,
            impurity_importance=true,
        )

        # Store results
        all_mdi_noisy[:, :, i] = impurity_importance(noisy_tree)
        all_mdi_signal[:, :, i] = impurity_importance(signal_tree)

        next!(p)
    end
    println("Simulation finished successfully.")

    # 3. Saving results
    mkpath(exp_dir)
    println("Saving results to `$exp_dir`...")

    # Define file paths
    json_path = joinpath(exp_dir, "params_$(exp_name).json")
    arrow_path = joinpath(exp_dir, "mdi_$(exp_name).arrow")
    dataset_path = joinpath(exp_dir, "dataset_$(exp_name).arrow")

    # Convert struct to dict for JSON saving
    pms_dict = Dict(string(field) => getfield(pms, field) for field in fieldnames(Hyperparameters))

    # Create dataframe for Arrow saving
    function create_results(mdi_array, tree_type, n_features)
        [(tree_type=tree_type, tree=t, feature=f, depth=d, mdi=mdi_array[f, d, t])
         for t in 1:pms.N_TREES, f in 1:n_features, d in 1:pms.MAX_DEPTH]
    end
    mdi_df = DataFrame(vcat(
        vec(create_results(all_mdi_noisy, "noisy", N_FEAT_TOTAL)),
        vec(create_results(all_mdi_signal, "signal", pms.N_FEAT_SIGNAL))
    ))

    # Create dataframe for the dataset
    dataset_df = DataFrame(hcat(X_noisy, y), :auto)
    rename!(dataset_df, names(dataset_df)[end] => :y)

    # Writing parameters and results to files
    open(json_path, "w") do f
        JSON.print(f, pms_dict, 4)
    end
    Arrow.write(arrow_path, mdi_df)
    Arrow.write(dataset_path, dataset_df)
    println("\nResults saved successfully.")
end