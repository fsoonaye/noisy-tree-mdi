using DataFrames
using Statistics
using Distributions
using Plots


"""
Compute mean, variance, and confidence interval for a sample using t-distribution.
"""
function monte_carlo(data::AbstractVector{<:Real}; ci_level::Real=0.95)
    n = length(data)
    μ = mean(data)
    std_err = std(data) / sqrt(n)
    ci_half_width = quantile(TDist(n - 1), 1 - (1 - ci_level) / 2) * std_err

    return (mean=μ, ci_lower=μ - ci_half_width, ci_upper=μ + ci_half_width)
end

"""
Process raw MDI data to compute cumulative MDI means and confidence intervals.
"""
function process_results(mdi_df::DataFrame, params::Hyperparameters)
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
            for feat_idx in 1:params.N_FEAT_SIGNAL
        )
    end

    noisy_results = extract_features(noisy_df)
    signal_results = extract_features(signal_df)

    return noisy_results, signal_results
end