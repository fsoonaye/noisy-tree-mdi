include("lib/sim.jl")

function main()
    # Define hyperparameters for the simulation
    pms = Hyperparameters(
        N_TREES=1_000,
        N_SAMPLES=100_000,
        N_FEAT_SIGNAL=3,
        N_FEAT_NOISY=3,
        MTRY=2,
        NODESIZE=5,
        MAX_DEPTH=15,
        RNG_SEED=123,
        EPS_STD=0.1,
        SIGNAL_COEFFS=[10.0, 5.0, 2.0],
    )

    # Validate parameters
    @assert 1 <= pms.MTRY <= pms.N_FEAT_SIGNAL
    @assert length(pms.SIGNAL_COEFFS) == pms.N_FEAT_SIGNAL

    run_sim(pms)
end

main()