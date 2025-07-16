using Plots
using LaTeXStrings

function plot_mdi(
    params::Hyperparameters;
    results_noisy::Union{Dict,Nothing}=nothing,
    results_signal::Union{Dict,Nothing}=nothing,
    feature_indices=1:params.N_FEAT_SIGNAL
)
    # Validate inputs
    if isnothing(results_noisy) && isnothing(results_signal)
        error("At least one of `results_noisy` or `results_signal` must be provided.")
    end

    x_values = 0:params.MAX_DEPTH

    # Helper function to add data to plot
    function add_data!(plt, data_dict, label, color, feat_idx)
        isnothing(data_dict) && return

        data = data_dict[feat_idx]
        error_lower = data.means .- data.ci_lower
        error_upper = data.ci_upper .- data.means

        plot!(plt, x_values, data.means,
            yerror=(error_lower, error_upper),
            label=label, color=color)
    end

    # Create subplot for each feature
    subplots = map(feature_indices) do feat_idx
        plt = plot(
            xlabel="Tree Depth",
            ylabel=L"MDI(X^{(%$feat_idx)})",
            title=L"X^{(%$feat_idx)}",
            grid=true,
            legend=:bottomright
        )

        add_data!(plt, results_signal, "Signal", :blue, feat_idx)
        add_data!(plt, results_noisy, "Noisy", :red, feat_idx)

        return plt
    end

    # Return single plot or vertical layout
    n_plots = length(subplots)
    if n_plots == 1
        return subplots[1]
    else
        return plot(subplots..., layout=(n_plots, 1), size=(600, 300 * n_plots))
    end
end