using HMM # Use the main module
using Test
using Random, Distributions, LinearAlgebra
using StatsBase
using LogExpFunctions

# Include simulation functions
include("test_utils.jl") 
import .TestUtils as SF



# --- Test Suite for MixtureHMM --- 

    # --- Common Data Simulation Setup ---
    K = 3  # Number of hidden states
    D = 2  # Number of mixture components
    N = 100 # Number of sequences (increase slightly for mixture?)
    T = 4
    SEED = 2 # Use a different seed

    sim_params_mix = SF.HMMRegressionSimulationParams(
        K = K,
        eta_dist = Normal(0, 1.5), # Override eta_sd
        T_range = (2, T)           # Set T_range based on test variable T
    )

    sim = SF.hmm_generate_multiple_eta_reg(
        seed = SEED, K = K, T_max = T, N = N, D = D, J = 1, 
        params = sim_params_mix     # Pass the custom params
    );
    
    y_rag, t_rag = SF.create_ragged_vector(sim.y, sim.T)
    
    # --- Get True Parameters (Sorted for comparison) --- 
    true_ω = sort(sim.θ.μ)
    true_η = sort(sim.θ.η) # Sorted true mixture means
    # Calculate true mixture weights based on simulation assignments
    η_counts = counts(sim.θ.η_id, 1:D)
    true_η_θ = η_counts ./ N 
    true_T_mat = sim.θ.A
    true_T_list = [true_T_mat[i, :] for i in 1:K]
    true_σ = sim.θ.σ

    mixture_data = MixtureHMMData(y_rag, K, D)
    params = initialize_params(MixtureHMMParams, SEED+2, mixture_data)

## Ed code starts

    log_u, log_nu, log_mix, log_z = ed_e_step(params, mixture_data);

    resp, gamma, xi = e_step(params, mixture_data);


    ed_m_step!(params, mixture_data, log_u, log_nu, log_mix, log_z)


    logdensity(params, mixture_data)



    ed_run_em!(params, mixture_data, maxiter=100, tol=1e-4)

    
    params.η_raw
    params.η_θ
    params.ω

    true_ω
    true_η

    hcat(sort(params.ω), sort(true_ω))
    hcat(sort(params.η_raw), sort(true_η))
    hcat(sort(params.η_θ), sort(true_η_θ))
    hcat(vcat(params.T_list...), vcat(true_T_list...))
    hcat(params.σ, true_σ)
    
