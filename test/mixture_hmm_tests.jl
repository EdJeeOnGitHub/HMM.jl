using HMM # Use the main module
using Test
using Random, Distributions, LinearAlgebra
using StatsBase

# Include simulation functions
include("test_utils.jl") 
import .TestUtils as SF



# --- Test Suite for MixtureHMM --- 
@testset "MixtureHMM Tests" begin

    # --- Common Data Simulation Setup ---
    K = 3  # Number of hidden states
    D = 2  # Number of mixture components
    N = 1000 # Number of sequences (increase slightly for mixture?)
    T = 20
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

    @testset "Mixture Initialization" begin
        mixture_data = MixtureHMMData(y_rag, K, D)
        params = initialize_params(MixtureHMMParams, SEED+1, mixture_data)
        
        @test params isa MixtureHMMParams{Float64}
        @test length(params.ω) == K
        @test length(params.η_raw) == D
        @test length(params.η_θ) == D
        @test sum(params.η_θ) ≈ 1.0
        @test length(params.T_list) == K
        @test all(sum(row) ≈ 1.0 for row in params.T_list)
        @test params.σ > 0
        
        @test isfinite(logdensity(params, mixture_data))
    end

    @testset "Mixture E-step and M-step" begin
        mixture_data = MixtureHMMData(y_rag, K, D)
        params = initialize_params(MixtureHMMParams, SEED+2, mixture_data)
        
        # Test e_step returns correct types and shapes
        r_nd, γ_dict, ξ_dict = e_step(params, mixture_data);
        @test r_nd isa Matrix{Float64}
        @test size(r_nd) == (N, D)

        @test all(sum(r_nd, dims=2) .≈ 1.0) # Responsibilities sum to 1 for each sequence

        @test γ_dict isa Dict
        @test ξ_dict isa Dict
        @test length(γ_dict) == N
        @test length(ξ_dict) == N
        # Check γ sums
        @test all( all(sum(γ_dict[i], dims=1) .≈ 1.0) for i in 1:N if size(γ_dict[i], 2) > 0 )
        # Check ξ sums
        all_xi_sums_approx_one = [ isapprox(sum(ξ_dict[i][:, :, t]), 1.0) for i in 1:N for t in 1:size(ξ_dict[i], 3) ]

        xi_sums = [dropdims(sum(ξ_dict[i], dims = [1, 2]), dims = (1, 2)) for i in 1:N]
        all_xi_approx_1 = [all(isapprox.(xi_sums[i], 1.0)) for i in 1:N]
        @test all(all_xi_approx_1)
        # Test m_step! modifies parameters
        old_params = deepcopy(params)
        m_step!(params, mixture_data, r_nd, γ_dict, ξ_dict)
        @test params != old_params
        # Check validity
        @test all(isapprox(sum(row), 1.0, atol=1e-6) for row in params.T_list)
        @test all(params.η_θ .>= 0) & (sum(params.η_θ) ≈ 1.0)
        @test params.σ > 0
    end



    @testset "Mixture EM Convergence and Recovery - near optimum" begin
        # Run EM using the parallel method dispatching on Type
        true_params  = MixtureHMMParams(
                true_η,
                true_η_θ,
                true_ω,
                true_T_list,
                true_σ
        )
        mixture_data = MixtureHMMData(y_rag, K, D)
        initial_params = initialize_params(MixtureHMMParams, SEED+1, mixture_data)
        initial_params.ω .= true_params.ω .+ randn(K) * 0.01
        initial_params.η_raw .= true_params.η_raw
        initial_params.η_θ .= true_params.η_θ
        initial_params.T_list .= true_params.T_list
        logdensity(initial_params, mixture_data)
        best_params = run_em!(initial_params, mixture_data, maxiter=1_000, tol=1e-4, verbose=false)



        
        @test best_params isa MixtureHMMParams{Float64}
        final_logp = logdensity(best_params, mixture_data)
        @test isfinite(final_logp)
        
        # --- Parameter Recovery ---
        # Sort estimated parameters for comparison
        est_η_perm = sortperm(best_params.η_raw)
        est_ω_perm = sortperm(best_params.ω)

        est_η_sorted = best_params.η_raw[est_η_perm]
        est_η_θ = best_params.η_θ
        est_ω_sorted = best_params.ω[est_ω_perm]
        est_T_list = best_params.T_list # Permute rows based on ω sorting
        est_T_mat = hcat(est_T_list...)' 

        # Compare sorted estimated params to sorted true params
        # Use loose tolerances, especially for beta
        hcat(est_η_sorted, true_η)
        hcat(sort(est_η_θ), sort(true_η_θ))
        hcat(est_ω_sorted, true_ω)


        @test est_ω_sorted ≈ true_ω rtol=0.1
        @test est_η_sorted ≈ true_η rtol=0.1
        @test sort(est_η_θ) ≈ sort(true_η_θ) rtol=0.1 # Check mixture weights

        # Compare transition matrices (might need looser tol or element-wise)
        # Note: True T matrix doesn't need sorting if states correspond to sorted ω
        @test est_T_mat ≈ true_T_mat rtol=0.1



        @test best_params.σ ≈ true_σ rtol=0.1

    end


    @testset "Mixture EM Convergence and Recovery" begin
        mixture_data = MixtureHMMData(y_rag, K, D)
        # Run EM using the parallel method dispatching on Type
        n_inits_test = 20 # Use a small number for faster tests
        

        best_params, results = run_em!(MixtureHMMParams, mixture_data; n_init=n_inits_test, maxiter=100, tol=1e-4, verbose=false);
        
        # Test the returned parameters
        @test best_params isa MixtureHMMParams{Float64}
        final_logp = logdensity(best_params, mixture_data)
        @test isfinite(final_logp)
        
        # Parameter recovery (tolerances might need significant adjustment)
        est_ω = sort(best_params.ω)
        est_η = sort(best_params.η_raw)
        hcat(est_ω, true_ω)
        hcat(est_η, true_η)
        @test all(isapprox.(est_ω, true_ω, atol=0.1))
        @test all(isapprox.(est_η, true_η, atol=1e-1))

        # Match estimated η_θ to true based on sorted η values
        sort_idx_est = sortperm(best_params.η_raw)
        # true_η is already sorted
        est_η_θ_sorted = best_params.η_θ[sort_idx_est]
        @test sort(est_η_θ_sorted) ≈ sort(true_η_θ) atol=0.1

        est_T_list = best_params.T_list
        for k in 1:K
            @test est_T_list[k] ≈ true_T_list[k] rtol=0.1
        end
        
        @test best_params.σ ≈ true_σ atol=0.05
    end

end # @testset MixtureHMM Tests 