using Parameters

@with_kw struct Hyperparameters
    N_TREES::Int
    N_SAMPLES::Int
    N_FEAT_SIGNAL::Int
    N_FEAT_NOISY::Int
    MTRY::Int
    NODESIZE::Int
    MAX_DEPTH::Int
    RNG_SEED::Int
    EPS_STD::Float64
    SIGNAL_COEFFS::Vector{Float64}
end