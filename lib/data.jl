using DataFrames
using Statistics
using Distributions
using Plots
using Arrow
using JSON

include("types.jl")

"""
Load simulation parameters, MDI results and dataset from JSON and Arrow files.
"""
function load_results(exp_name::String; results_dir="results")
    exp_dir = joinpath(results_dir, exp_name)
    json_path = joinpath(exp_dir, "params_$(exp_name).json")
    arrow_path = joinpath(exp_dir, "mdi_$(exp_name).arrow")
    dataset_path = joinpath(exp_dir, "dataset_$(exp_name).arrow")

    if !isdir(exp_dir)
        error("Experiment directory not found: `$exp_dir`")
    end

    if !isfile(json_path) || !isfile(arrow_path) || !isfile(dataset_path)
        error("Result files not found in `$exp_dir` directory.")
    end

    # Load parameters and convert to struct
    pms_dict = JSON.parsefile(json_path)
    pms = Hyperparameters(; (Symbol(k) => v for (k, v) in pms_dict)...)

    # Load MDI data and dataset
    mdi_df = DataFrame(Arrow.Table(arrow_path))
    dataset_df = DataFrame(Arrow.Table(dataset_path))

    return pms, mdi_df, dataset_df
end


"""
Compute mean, variance, and confidence interval for a sample using t-distribution.
"""
function monte_carlo(data::AbstractVector{<:Real}; α::Real=0.05)
    n = length(data)
    μ = mean(data)
    std_err = std(data) / sqrt(n)
    ci_half_width = quantile(TDist(n - 1), 1 - α / 2) * std_err
    return (mean=μ, ci_lower=μ - ci_half_width, ci_upper=μ + ci_half_width)
end

"""
Process raw MDI data to compute cumulative MDI means and confidence intervals.
"""
function process_results(mdi_df::DataFrame, pms::Hyperparameters)
    # Calculate cumulative MDI
    transform!(groupby(mdi_df, [:tree_type, :tree, :feature]), :mdi => cumsum => :cumulative_mdi)

    # Calculate monte carlo statistics
    stats_df = combine(groupby(mdi_df, [:tree_type, :feature, :depth]),
        :cumulative_mdi => monte_carlo => AsTable)

    # Extract results by tree type
    noisy_df = filter(r -> r.tree_type == "noisy", stats_df)
    signal_df = filter(r -> r.tree_type == "signal", stats_df)

    function extract_features(df)
        Dict(
            feat_idx => begin
                feat_data = filter(r -> r.feature == feat_idx, df)

                # Prepend depth 0 with zeros
                (means=[0.0; feat_data.mean],
                    ci_lower=[0.0; feat_data.ci_lower],
                    ci_upper=[0.0; feat_data.ci_upper])
            end
            for feat_idx in 1:pms.N_FEAT_SIGNAL
        )
    end

    noisy_results = extract_features(noisy_df)
    signal_results = extract_features(signal_df)

    return noisy_results, signal_results
end