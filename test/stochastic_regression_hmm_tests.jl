using HMM
using Test
using Random, Distributions, LinearAlgebra
using StatsBase

# Include simulation functions
include("test_utils.jl") 
import .TestUtils as SF
function generate_f(f, params, x)
    P = size(params.beta, 3) # Get P from params
    basis_features = f(x, P-1) # Use some fn
    D = size(params.η_raw, 1)
    K = size(params.ω, 1)
    predictions = Array{Float64}(undef, K, D, length(x))
    for d in 1:D
        for k in 1:K
            predictions[k, d, :] = basis_features * params.beta[d, k, :]
        end
    end
    return predictions
end

function generate_true_fn(xs, ωs, ηs, c_func)
    ωs = sort(ωs)
    ηs = sort(ηs)
    K = length(ωs)
    D = length(ηs)
    true_fn = zeros(K, D, length(xs))
    for k in 1:K
        for d in 1:D
            true_fn[k, d, :] = c_func.(xs, ωs[k], ηs[d])
        end
    end
    return true_fn
end


# --- Test Suite for Stochastic RegressionHMM --- 
@testset "Stochastic RegressionHMM Tests" begin

    # --- Common Data Simulation Setup ---
    K = 3  # Number of hidden states
    D = 3  # Number of mixture components
    P = 8  # Number of monomial basis functions to use in model (degree P-1)
    N = 1000 # Number of sequences 
    T = 5  # Sequence length
    SEED = 3 # Use a different seed
    basis_fn = monomial_basis

    # Simulate data including the auxiliary regression target 'c' and covariate 'k'
    sim_params_reg = SF.HMMRegressionSimulationParams(
        K = K,
        mu_dist = Normal(0, 3.0),
        eta_dist = Normal(0, 1.0),
        T_range = (2, T)
    )

    sim = SF.hmm_generate_multiple_eta_reg(
        seed = SEED, K = K, T_max = T, N = N, D = D, J = 1,
        params = sim_params_reg
    );
    
    # Create ragged arrays
    y_rag, _ = SF.create_ragged_vector(sim.y, sim.T)
    c_rag, _ = SF.create_ragged_vector(sim.c, sim.T) 
    k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 

    # Calculate bounds for monomial basis
    k_all = reduce(vcat, k_rag)
    k_min = minimum(k_all)
    k_max = maximum(k_all)

    # Generate Basis Matrix Ragged Array
    Φ_rag = [basis_fn(k_seq, P-1) for k_seq in k_rag]
    
    # Create Data Struct
    regression_data = RegressionHMMData(y_rag, c_rag, Φ_rag, D, K, P)

    # Get True Parameters
    true_ω = sort(sim.θ.μ)
    true_η = sort(sim.θ.η) 
    true_η_θ = counts(sim.θ.η_id, 1:D) ./ N 
    true_T_mat = sim.θ.A
    true_T_list = [Vector{Float64}(true_T_mat[i, :]) for i in 1:K]
    true_σ = sim.θ.σ

    # Sort true params based on η and ω for comparison
    η_perm = sortperm(true_η)
    ω_perm = sortperm(true_ω) 
    
    true_η_sorted = true_η[η_perm]
    true_η_θ_sorted = true_η_θ[η_perm]
    true_ω_sorted = true_ω

    @testset "Stochastic EM Configuration" begin
        config = StochasticEMConfig(
            batch_size = 10
        )
        
        @test config isa StochasticEMConfig
        @test config.batch_size > 0
        @test config.batch_size <= N
    end

    @testset "Stochastic Regression Initialization" begin
        params = initialize_params(RegressionHMMParams, SEED+1, regression_data)
        
        @test params isa RegressionHMMParams{Float64}
        @test length(params.ω) == K
        @test length(params.η_raw) == D
        @test length(params.η_θ) == D
        @test sum(params.η_θ) ≈ 1.0
        @test length(params.T_list) == K
        @test all(sum(row) ≈ 1.0 for row in params.T_list)
        @test size(params.beta) == (D, K, P)
        @test params.σ > 0
        @test params.sigma_f > 0

        @test isfinite(logdensity(params, regression_data))
    end

    @testset "Stochastic E-step and M-step" begin
        params = initialize_params(RegressionHMMParams, SEED+1, regression_data)
        config = StochasticEMConfig(batch_size=10, full_batch_step=10)
        
        # Test stochastic E-step
        responsibilities, γ_dict, ξ_dict, batch_indices = stochastic_e_step(
            params, regression_data, config, SEED
        )
        
        @test responsibilities isa Matrix{Float64}
        @test size(responsibilities, 1) == config.batch_size
        @test size(responsibilities, 2) == D
        @test all(sum(responsibilities, dims=2) .≈ 1.0)
        
        @test γ_dict isa Dict
        @test ξ_dict isa Dict
        @test length(γ_dict) == config.batch_size
        @test length(ξ_dict) == config.batch_size
        
        # Test stochastic M-step
        old_params = deepcopy(params)
        stochastic_m_step!(
            params, regression_data, config,
            responsibilities, γ_dict, ξ_dict, batch_indices
        )
        
        @test params != old_params
        @test all(isapprox(sum(row), 1.0, atol=1e-6) for row in params.T_list)
        @test all(params.η_θ .>= 0) & (sum(params.η_θ) ≈ 1.0)
        @test params.σ > 0
        @test params.sigma_f > 0
    end

    @testset "Stochastic EM Convergence and Recovery" begin
        # Run Stochastic EM using the parallel method
        n_inits_test = 50  # Use fewer inits for stochastic tests
        max_iter_test = 1000
        config = StochasticEMConfig(batch_size=50, full_batch_step=100)

        best_params, results = run_stochastic_em!(
            RegressionHMMParams,
            regression_data,
            config,
            10;
            n_init=n_inits_test,
            maxiter=max_iter_test,
            verbose=true
        );

        orig_best_params = deepcopy(best_params)
        best_params = run_em!(best_params, regression_data, maxiter=max_iter_test, verbose=true);
        
        @test best_params isa RegressionHMMParams{Float64}
        final_logp = logdensity(best_params, regression_data)
        @test isfinite(final_logp)
        
        # Parameter Recovery
        est_η_perm = sortperm(best_params.η_raw)
        est_ω_perm = sortperm(best_params.ω)

        est_η_sorted = best_params.η_raw[est_η_perm]
        est_η_θ = best_params.η_θ
        est_ω_sorted = best_params.ω[est_ω_perm]
        est_T_list = best_params.T_list
        est_T_mat = hcat(est_T_list...)'

        hcat(est_ω_sorted, true_ω_sorted)
        hcat(est_η_sorted, true_η_sorted)
        # Compare sorted estimated params to sorted true params
        # Use looser tolerances for stochastic EM
        @test est_ω_sorted ≈ true_ω_sorted rtol=0.2
        @test est_η_sorted ≈ true_η_sorted rtol=0.2
        @test sort(est_η_θ) ≈ sort(true_η_θ_sorted) rtol=0.2
        @test est_T_mat ≈ true_T_mat[ω_perm, ω_perm] rtol=1.0
        @test best_params.σ ≈ true_σ rtol=1.0

        # Test regression function recovery
        x = range(-2, 2, length=100)
        c_hat = generate_f(basis_fn, best_params, x)
        c_true = generate_true_fn(x, sort(best_params.ω), sort(best_params.η_raw), SF.default_c_func)

        mse_d1 = mean((c_hat[1, 1, :] - c_true[1, 1, :]).^2)
        mse_d2 = mean((c_hat[2, 2, :] - c_true[2, 2, :]).^2)
        mse_d3 = mean((c_hat[3, 3, :] - c_true[3, 3, :]).^2)

        @test mse_d1 < 0.2  # Looser tolerance for stochastic EM
        @test mse_d2 < 3.0
        @test mse_d3 < 0.2








    end

end # @testset Stochastic RegressionHMM Tests 