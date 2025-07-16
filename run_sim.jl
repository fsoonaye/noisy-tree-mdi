include("lib/sim.jl")

function main()
    params = Hyperparameters(
        N_TREES=1000,
        N_SAMPLES=200000,
        N_FEAT_SIGNAL=3,
        N_FEAT_NOISY=2,
        SIGNAL_COEFFS=[-81.0, 31.0, 14.0],
        EPS_STD=0.1,
        MAX_DEPTH=15,
        MTRY=2,
        NODESIZE=5,
        RNG_SEED=123,
    )

    # Validate parameters
    @assert 1 <= params.MTRY <= params.N_FEAT_SIGNAL
    @assert length(params.SIGNAL_COEFFS) == params.N_FEAT_SIGNAL

    # Run the simulation
    run_sim(params)
end

main()