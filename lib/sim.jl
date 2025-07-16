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

@with_kw struct Hyperparameters
    N_TREES::Int
    N_SAMPLES::Int
    N_FEAT_SIGNAL::Int
    N_FEAT_NOISY::Int
    SIGNAL_COEFFS::Vector{Float64}
    EPS_STD::Float64
    MAX_DEPTH::Int
    MTRY::Int
    NODESIZE::Int
    RNG_SEED::Int
end

"""
Generate synthetic dataset with signal and noise features based on a linear model.
"""
function gen_linear_data(
    n_samples::Int,
    signal_coeffs::Vector{Float64},
    n_noise_features::Int,
    noise_std::Float64,
    rng::Random.AbstractRNG,
)
    n_signal = length(signal_coeffs)

    # Generate signal features (uniform distribution)
    x_signal = rand(rng, n_samples, n_signal)

    # Generate noise features (uniform distribution)
    z_noise = rand(rng, n_samples, n_noise_features)

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
function params_to_filename(params::Hyperparameters)
    parts = [
        "d$(params.N_FEAT_SIGNAL)",
        "p$(params.N_FEAT_NOISY)",
        "m$(params.MTRY)",
        "k$(params.MAX_DEPTH)",
        "n$(params.N_SAMPLES)",
        "M$(params.N_TREES)",
    ]
    return join(parts, "_")
end


function run_sim(params::Hyperparameters)
    # 1. Setup and print parameters
    println("Initializing Simulation...")
    println("Parameters:")
    for field in fieldnames(Hyperparameters)
        println("    - ", rpad(string(field), 15), ": ", getfield(params, field))
    end
    println("    - Threads        : ", Threads.nthreads())
    println("\n")

    N_FEAT_TOTAL = params.N_FEAT_SIGNAL + params.N_FEAT_NOISY
    master_rng = MersenneTwister(params.RNG_SEED)
    all_mdi_noisy = zeros(N_FEAT_TOTAL, params.MAX_DEPTH, params.N_TREES)
    all_mdi_signal = zeros(params.N_FEAT_SIGNAL, params.MAX_DEPTH, params.N_TREES)

    # Generate synthetic data once for all runs
    X_signal, X_noisy, y = gen_linear_data(
        params.N_SAMPLES, params.SIGNAL_COEFFS, params.N_FEAT_NOISY, params.EPS_STD, master_rng
    )

    # 2. Fitting all trees
    p = Progress(params.N_TREES, "Fitting $(params.N_TREES) noisy and signal trees:")

    master_seed = rand(master_rng, UInt)
    @time Threads.@threads for i in 1:params.N_TREES
        # Create a thread-safe RNG for each iteration
        local_rng = MersenneTwister(master_seed + i)

        # Fit noisy tree
        noisy_tree = build_tree(
            y,
            X_noisy,
            params.MTRY,
            params.MAX_DEPTH,
            params.NODESIZE;
            rng=local_rng,
            impurity_importance=true,
        )

        # Fit signal tree
        signal_tree = build_tree(
            y,
            X_signal,
            params.MTRY,
            params.MAX_DEPTH,
            params.NODESIZE;
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
    base_filename = params_to_filename(params)
    results_dir = "results"
    mkpath(results_dir)

    arrow_path = joinpath(results_dir, base_filename * ".arrow")
    json_path = joinpath(results_dir, base_filename * ".json")

    # Check if files already exist to prevent accidental overwrites
    if isfile(arrow_path) || isfile(json_path)
        error(
            "Results files already exist for this parameter set:\n" *
            "  - $arrow_path\n" *
            "  - $json_path\n" *
            "Please move or delete them before re-running.",
        )
    end

    println("Saving results to `$arrow_path` and `$json_path`...")

    # Convert struct to dict for JSON saving
    params_dict = Dict(string(field) => getfield(params, field) for field in fieldnames(Hyperparameters))

    # Create dataframe for Arrow saving
    function create_results(mdi_array, tree_type)
        [(tree_type=tree_type, tree=t, feature=f, depth=d, mdi=mdi_array[f, d, t])
         for t in 1:params.N_TREES, f in 1:params.N_FEAT_SIGNAL, d in 1:params.MAX_DEPTH]
    end
    df = DataFrame(vcat(
        create_results(all_mdi_noisy, "noisy"),
        create_results(all_mdi_signal, "signal")
    ))

    # Writing parameters and results to files
    open(json_path, "w") do f
        JSON.print(f, params_dict, 4)
    end
    Arrow.write(arrow_path, df)
    println("\nResults saved successfully.")
end

"""
Load simulation parameters and MDI results from JSON and Arrow files.
"""
function load_results(base_filename::String; results_dir="results")
    json_path = joinpath(results_dir, base_filename * ".json")
    arrow_path = joinpath(results_dir, base_filename * ".arrow")

    if !isfile(json_path) || !isfile(arrow_path)
        error("Result files not found for $base_filename in `$results_dir` directory.")
    end

    # Load parameters and convert to struct
    params_dict = JSON.parsefile(json_path)
    params = Hyperparameters(; (Symbol(k) => v for (k, v) in params_dict)...)

    # Load MDI data
    mdi_df = DataFrame(Arrow.Table(arrow_path))

    return params, mdi_df
end