include("lib/sim.jl")
include("lib/viz.jl")
include("lib/data.jl")

function main()
    # Read experiment name from CLI
    if isempty(ARGS)
        println("Usage: julia view_sim.jl <experiment_name>")
        println("Example: julia view_sim.jl d3_p2_m2_k15_n200000_M1000")
        return
    end
    exp_name = ARGS[1]

    println("1. Loading results for `$exp_name`...")
    local pms, mdi_df, dataset_df
    try
        pms, mdi_df, dataset_df = load_results(exp_name)
    catch e
        if isa(e, ErrorException) && contains(e.msg, "not found")
            println("\nError: Could not find results for `$exp_name`.")
            println("Make sure the files exist and that the experiment name is correctely spelled.")
            return
        else
            rethrow()
        end
    end
    println("   ...done.")

    println("2. Processing results...")
    results_noisy, results_signal = process_results(mdi_df, pms)
    println("   ...done.")

    println("3. Generating plot...")
    fig = plot_mdi(
        pms;
        results_noisy=results_noisy,
        results_signal=results_signal
    )
    println("   ...done.")

    display(fig)

    # Define the directory for saving plots
    plot_dir = joinpath("plots", exp_name)
    mkpath(plot_dir)

    # Find the next available version number for the plot
    version = 1
    while true
        save_path = joinpath(plot_dir, "plot_v$(version)_$(exp_name).pdf")
        if !isfile(save_path)
            savefig(fig, save_path)
            println("Plot saved to `$save_path`")
            break
        end
        version += 1
    end
end

main()