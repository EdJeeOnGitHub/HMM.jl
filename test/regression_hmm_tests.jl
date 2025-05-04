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

    function generate_true_fn(true_params, x)
        D = size(true_params.η, 1)
        K = size(true_params.μ, 1)
        true_fn = Array{Float64}(undef, K, D, length(x))
        η_sorted = sort(true_params.η)
        for d in 1:D
            for k in 1:K
                if k == 1
                    true_fn[k, d, :] = x .+ η_sorted[d]
                elseif k == 2
                    true_fn[k, d, :] = -x .+ η_sorted[d]
                elseif k > 2
                    true_fn[k, d, :] = sin.(x * k) .+ η_sorted[d]
                end
            end
        end
        return true_fn
    end

# --- Test Suite for RegressionHMM (Mixture Regression Variant) --- 
@testset "RegressionHMM Tests" begin

    # --- Common Data Simulation Setup ---
    K = 3  # Number of hidden states
    D = 3  # Number of mixture components
    P = 8     # Number of monomial basis functions to use in model (degree P-1)
    N = 100 # Number of sequences 
    T = 25  # Sequence length
    SEED = 3 # Use a different seed

    # Simulate data including the auxiliary regression target 'c' and covariate 'k'
    # Note: Simulation uses M_true for generating the underlying function.
    # The model will fit using P basis functions.
    sim_params_reg = SF.HMMRegressionSimulationParams(
        K = K,
        mu_dist = Normal(0, 3.0),  # Override mu_sd
        eta_dist = Normal(0, 1.0), # Override eta_sd
        T_range = (2, T)           # Set T_range based on test variable T
    )

    sim = SF.hmm_generate_multiple_eta_reg(
        seed = SEED, K = K, T_max = T, N = N, D = D, J = 1,
        params = sim_params_reg     # Pass the custom params
    );
    
    # Create ragged arrays
    y_rag, _ = SF.create_ragged_vector(sim.y, sim.T)
    c_rag, _ = SF.create_ragged_vector(sim.c, sim.T) 
    k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 

    # --- Calculate bounds for monomial basis --- 
    k_all = reduce(vcat, k_rag)
    k_min = minimum(k_all)
    k_max = maximum(k_all)

    # --- Generate Basis Matrix Ragged Array ---
    Φ_rag = [monomial_basis(k_seq, P-1) for k_seq in k_rag]
    
    # --- Create Data Struct ---
    # Use Φ_rag and P instead of k_rag and M
    regression_data = RegressionHMMData(y_rag, c_rag, Φ_rag, D, K, P)

    # --- Get True Parameters --- 
    # Note: True beta parameters from simulation might not be directly comparable 
    # if M_true != P-1, as the basis is different.
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
    true_ω_sorted = true_ω # Already sorted above


    @testset "Regression Initialization" begin
        params = initialize_params(RegressionHMMParams, SEED+1, regression_data);
        
        @test params isa RegressionHMMParams{Float64}
        @test length(params.ω) == K
        @test length(params.η_raw) == D
        @test length(params.η_θ) == D
        @test sum(params.η_θ) ≈ 1.0
        @test length(params.T_list) == K
        @test all(sum(row) ≈ 1.0 for row in params.T_list)
        @test size(params.beta) == (D, K, P) # Check against P
        @test params.σ > 0
        @test params.sigma_f > 0

        logdensity(params, regression_data)

        @test isfinite(logdensity(params, regression_data))
    end

    @testset "Regression E-step and M-step" begin
        params = initialize_params(RegressionHMMParams, SEED+1, regression_data)
        
        # Test e_step (assuming signature: e_step(params, data))
        # It should return responsibilities, gamma, xi based on the mixture regression logic
        r_nd, γ_dict, ξ_dict = e_step(params, regression_data) 
        @test r_nd isa Matrix{Float64}
        @test size(r_nd) == (N, D)
        @test all(sum(r_nd, dims=2) .≈ 1.0) # Responsibilities sum to 1

        @test γ_dict isa Dict
        @test ξ_dict isa Dict
        @test length(γ_dict) == N
        @test length(ξ_dict) == N
        @test all( all(sum(γ_dict[i], dims=1) .≈ 1.0) for i in 1:N if size(γ_dict[i], 2) > 0 )
        all_xi = [ sum(ξ_dict[i][:, :, t]) for i in 1:N for t in 1:size(ξ_dict[i], 3) ]

        @test all(isapprox.(all_xi, 1.0, atol=1e-6))
        # Test m_step! (assuming signature: m_step!(params, data, r_nd, γ_dict, ξ_dict))
        old_params_ca = deepcopy(params)
        m_step!(params, regression_data, r_nd, γ_dict, ξ_dict)
        @test params != old_params_ca
        # Check validity after M-step
        @test all(isapprox(sum(row), 1.0, atol=1e-6) for row in params.T_list)
        @test all(params.η_θ .>= 0) & (sum(params.η_θ) ≈ 1.0)
        @test params.σ > 0
        @test params.sigma_f > 0
    end

    @testset "Regression EM Convergence and Recovery - near optimum" begin
        # Run EM using the parallel method dispatching on Type
        true_params  = RegressionHMMParams(
            true_η,
            true_η_θ,
            true_ω,
            true_T_list,
            true_σ,
            fill(0.1, D, K, P), # Use P here
            0.1
        )
        initial_params = initialize_params(RegressionHMMParams, SEED+1, regression_data)
        initial_params.ω .= true_params.ω
        initial_params.η_raw .= true_params.η_raw
        initial_params.η_θ .= true_params.η_θ
        initial_params.T_list .= true_params.T_list
        logdensity(initial_params, regression_data)
        best_params = run_em!(initial_params, regression_data, maxiter=1_000, tol=1e-4, verbose=false)



        
        @test best_params isa RegressionHMMParams{Float64}
        final_logp = logdensity(best_params, regression_data)
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
        hcat(est_η_sorted, true_η_sorted)
        hcat(sort(est_η_θ), sort(true_η_θ))
        hcat(est_ω_sorted, true_ω_sorted)


        @test est_ω_sorted ≈ true_ω_sorted rtol=0.1
        @test est_η_sorted ≈ true_η_sorted rtol=0.1
        @test sort(est_η_θ) ≈ sort(true_η_θ_sorted) rtol=0.1 # Check mixture weights

        # Compare transition matrices (might need looser tol or element-wise)
        # Note: True T matrix doesn't need sorting if states correspond to sorted ω
        @test est_T_mat ≈ true_T_mat[ω_perm, ω_perm] rtol=1.0



        @test best_params.σ ≈ true_σ rtol=1.0
    x = range(-1, 1, length=100)
    


    c_hat = generate_f(monomial_basis, best_params, x)
    c_true = generate_true_fn(sim.θ, x)

    mse_d1 = mean((c_hat[1, 1, :] - c_true[1, 1, :]).^2)
    mse_d2 = mean((c_hat[2, 2, :] - c_true[2, 2, :]).^2)
    mse_d3 = mean((c_hat[3, 3, :] - c_true[3, 3, :]).^2)


    @test mse_d1 < 0.1
    @test mse_d2 < 0.1
    @test mse_d3 < 0.1


# using Plots
#    p_list = []
#    for d in 1:D
#     for k in 1:K
#         p = plot(
#             x, c_hat[k, d, :], label="Predicted",
#         )
#         plot!(p, x, c_true[k, d, :], label="True", linestyle = :dash)
#         push!(p_list, p)
#     end
#    end
#    plot(p_list..., layout=(D, K))


    end

    @testset "Regression EM Convergence and Recovery" begin
        # Run EM using the parallel method dispatching on Type
        n_inits_test = 10 # Use fewer inits for regression tests (can be slow)
        max_iter_test = 10

        best_params, results = run_em!(RegressionHMMParams, regression_data; 
                                        n_init=n_inits_test, 
                                        maxiter=max_iter_test, 
                                        tol=1e-4, # Looser tolerance for regression
                                        verbose=false); 
        
        @test best_params isa RegressionHMMParams{Float64}
        final_logp = logdensity(best_params, regression_data)
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
        hcat(est_η_sorted, true_η_sorted)
        hcat(sort(est_η_θ), sort(true_η_θ))
        hcat(est_ω_sorted, true_ω_sorted)


        @test est_ω_sorted ≈ true_ω_sorted rtol=0.1
        @test est_η_sorted ≈ true_η_sorted rtol=0.1
        @test sort(est_η_θ) ≈ sort(true_η_θ_sorted) rtol=0.1 # Check mixture weights

        # Compare transition matrices (might need looser tol or element-wise)
        # Note: True T matrix doesn't need sorting if states correspond to sorted ω
        @test est_T_mat ≈ true_T_mat[ω_perm, ω_perm] rtol=0.5



        @test best_params.σ ≈ true_σ rtol=0.1

    x = range(-1, 1, length=100)
    c_hat = generate_f(monomial_basis, best_params, x)
    c_true = generate_true_fn(sim.θ, x)

    mse_d1 = mean((c_hat[1, 1, :] - c_true[1, 1, :]).^2)
    mse_d2 = mean((c_hat[2, 2, :] - c_true[2, 2, :]).^2)
    mse_d3 = mean((c_hat[3, 3, :] - c_true[3, 3, :]).^2)

    @test mse_d1 < 0.01
    @test mse_d2 < 0.01
    @test mse_d3 < 0.1



#    p_list = []
#    for d in 1:D
#     for k in 1:K
#         p = plot(
#             x, c_hat[k, d, :], label="Predicted",
#         )
#         plot!(p, x, c_true[k, d, :], label="True", linestyle = :dash)
#         push!(p_list, p)
#     end
#    end
#    plot(p_list..., layout=(D, K))


    end

end # @testset RegressionHMM Tests 