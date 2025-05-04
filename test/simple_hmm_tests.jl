using HMM # Use the main module
using Test
using Random, Distributions, LinearAlgebra
# using DataFrames # Optional: Keep if you want to use tidy_sim_data for comparisons

# Include simulation functions (assuming they are in test_utils.jl)
include("test_utils.jl") 
import .TestUtils as SF # Use the alias matching simple-em.jl for consistency


K = 3
T = 2
N = 500

# Define params specifically for this initial data generation
simple_test_params = SF.HMMRegressionSimulationParams(
    K = K,
    eta_dist = Normal(0, 0.0) # Override eta_sd=0
)

test_data = SF.hmm_generate_multiple_eta_reg(
    seed = 1, K = K, T_max = T, N = N, D = 1, J = 1, 
    params = simple_test_params # Pass the custom params
);

y = test_data.y
T = test_data.T
y_rag, t_rag = SF.create_ragged_vector(y, T)
k_rag, _ = SF.create_ragged_vector(test_data.k, T)
c_rag, _ = SF.create_ragged_vector(test_data.c, T)



# --- Test Suite for SimpleHMM --- 
@testset "SimpleHMM Tests" begin

    # --- Common Data Simulation Setup for all tests in this set ---
    K = 3
    N = 1000 # Smaller N for faster tests
    T = 20
    SEED = 1

    # Define params for the main simulation within the testset
    sim_params = SF.HMMRegressionSimulationParams(
        K = K,
        eta_dist = Normal(0, 0.0), # Override eta_sd=0
        T_range = (2, T) # Ensure max T matches the T variable here
    )

    sim = SF.hmm_generate_multiple_eta_reg(
        seed = SEED, K = K, T_max = T, N = N, D = 1, J = 1, 
        params = sim_params # Pass the custom params
    );
    
    y_rag, t_rag = SF.create_ragged_vector(sim.y, sim.T) # Use sim.T which is T_lengths
    
    true_ω = sort(sim.θ.μ) # Ensure comparison is with sorted means
    true_T_mat = sim.θ.A
    true_T_list = [true_T_mat[i, :] for i in 1:K]
    # true_σ = sim.θ.σ # Can compare sigma now

    # --- Individual Test Sets ---

    @testset "Initialization" begin
        hmm_data = SimpleHMMData(y_rag, K)
        params = initialize_params(SimpleHMMParams, SEED+1, hmm_data) # Use different seed for init
        
        @test params isa SimpleHMMParams{Float64} # Check type
        @test length(params.ω) == K
        @test length(params.T_list) == K
        @test all(length(row) == K for row in params.T_list)
        @test all(sum(row) ≈ 1.0 for row in params.T_list)
        @test params.σ > 0
        
        # Check logdensity is finite with initial params
        @test isfinite(logdensity(params, hmm_data))
    end

    @testset "E-step and M-step" begin
        hmm_data = SimpleHMMData(y_rag, K)
        params = initialize_params(SimpleHMMParams, SEED+2, hmm_data)

        # Test e_step returns dictionaries with correct structure
        γ_dict, ξ_dict = e_step(params, hmm_data)
        @test γ_dict isa Dict
        @test ξ_dict isa Dict
        @test length(γ_dict) == N
        @test length(ξ_dict) == N
        
        # Check γ probabilities sum to 1
        # Need to handle potential Float32/Float64 issues if types differ
        @test all( all(sum(γ_dict[i], dims = 1) .≈ 1.0) for i in 1:N if size(γ_dict[i], 2) > 0 ) 

        # Check ξ probabilities sum to 1 (across all sequences and time steps)
        all_xi_sums_approx_one = [ isapprox(sum(ξ_dict[i][:, :, t]), 1.0) for i in 1:N for t in 1:size(ξ_dict[i], 3) ]
        @test all(all_xi_sums_approx_one)

        # Test m_step! modifies parameters
        m_step!(params, hmm_data, γ_dict, ξ_dict)
        
        # Check basic validity of updated params
        @test all(isapprox(sum(row), 1.0) for row in params.T_list)
        @test params.σ > 0
    end

    @testset "EM Convergence and Parameter Recovery" begin
        hmm_data = SimpleHMMData(y_rag, K)
        params = initialize_params(SimpleHMMParams, SEED+3, hmm_data)
        initial_logp = logdensity(params, hmm_data)
        
        # Run EM (suppress verbose output during tests)
        run_em!(params, hmm_data; maxiter=50, tol=1e-4, verbose=false)
        final_logp = logdensity(params, hmm_data)

        @test isfinite(final_logp)
        @test final_logp > initial_logp # Expect improvement

        # Parameter recovery tests (might need tolerance adjustments)
        # Sort estimated omega for comparison
        est_ω = sort(params.ω)
        @test est_ω ≈ true_ω atol=0.5 # Adjust tolerance as needed

        
        # Compare sigma (if true value is known/relevant)
        @test params.σ ≈ sim.θ.σ atol=0.2 # Compare sigma now
    end

end # @testset SimpleHMM Tests 