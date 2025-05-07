using HMM
using BenchmarkTools
using Random
using LinearAlgebra # Likely needed for setup
using Distributions # Likely needed for setup

# Assuming test_utils.jl is in ../test/
println("Including test utils from: ", joinpath(@__DIR__, "..", "test", "test_utils.jl"))
include(joinpath(@__DIR__, "..", "test", "test_utils.jl"))
import .TestUtils as SF

# Define the main benchmark suite
const SUITE = BenchmarkGroup()

# Add a group for RegressionHMM benchmarks
SUITE["RegressionHMM"] = BenchmarkGroup()

# --- Baseline Scenario ---
let N=50, T=25, K=3, D=3, P=4, M_true=5, SEED=123, maxiter_bm=10
    SUITE["RegressionHMM"]["Baseline N=$N T=$T K=$K D=$D P=$P"] = @benchmarkable(
        run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
        setup = begin
            Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T, N = $N, D = $D, J = $M_true + 1, 
                params = sim_params
            );

            # Create ragged arrays
            y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
            c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
            k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
            # --- Generate Basis Matrix Ragged Array ---
            Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
            # --- Create Data Struct ---
            # Use Φ_rag and P instead of k_rag and M
            data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P)
            params = initialize_params(RegressionHMMParams, $SEED + 1, data)
        end,
        evals = 1 # Evaluate setup code only once per sample
    )
end

# --- Scalability: Number of Sequences (N) ---
SUITE["RegressionHMM"]["Scalability N"] = BenchmarkGroup()
let T=25, K=3, D=3, P=4, M_true=5, SEED=123, maxiter_bm=10
    for N_val in [10, 100, 500]
        SUITE["RegressionHMM"]["Scalability N"]["N=$N_val"] = @benchmarkable(
            run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
            setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T, N = $N_val, D = $D, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
            end,
            evals = 1
        )
    end
end

# --- Scalability: Sequence Length (T) ---
SUITE["RegressionHMM"]["Scalability T"] = BenchmarkGroup()
let N=50, K=3, D=3, P=4, M_true=5, SEED=123, maxiter_bm=10
    for T_val in [25, 100, 250]
        SUITE["RegressionHMM"]["Scalability T"]["T=$T_val"] = @benchmarkable(
            run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
            setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T_val, N = $N, D = $D, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
            end,
            evals = 1
        )
    end
end

# --- Scalability: Number of States (K) ---
SUITE["RegressionHMM"]["Scalability K"] = BenchmarkGroup()
let N=50, T=25, D=3, P=4, M_true=5, SEED=123, maxiter_bm=10
    for K_val in [2, 5, 10]
        SUITE["RegressionHMM"]["Scalability K"]["K=$K_val"] = @benchmarkable(
            run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
            setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K_val,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K_val, T_max = $T, N = $N, D = $D, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K_val, $P)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
            end,
            evals = 1
        )
    end
end

# --- Scalability: Number of Mixture Components (D) ---
SUITE["RegressionHMM"]["Scalability D"] = BenchmarkGroup()
let N=50, T=25, K=3, P=4, M_true=5, SEED=123, maxiter_bm=10
    for D_val in [2, 3, 5]
        SUITE["RegressionHMM"]["Scalability D"]["D=$D_val"] = @benchmarkable(
            run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
            setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T, N = $N, D = $D_val, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D_val, $K, $P)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
            end,
            evals = 1
        )
    end
end

# --- Scalability: Number of Basis Functions (P) ---
SUITE["RegressionHMM"]["Scalability P"] = BenchmarkGroup()
let N=50, T=25, K=3, D=3, M_true=5, SEED=123, maxiter_bm=10
    # Note: M_true (used for simulation generation) is kept constant.
    # We are varying the number of basis functions P used by the *model*.
    for P_val in [4, 8, 16]
        SUITE["RegressionHMM"]["Scalability P"]["P=$P_val"] = @benchmarkable(
            run_em!(params, data; maxiter=$maxiter_bm, verbose=false),
            setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T, N = $N, D = $D, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P_val-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P_val)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
            end,
            evals = 1
        )
    end
end

# --- Parallel Execution ---
SUITE["RegressionHMM"]["Parallel Execution"] = BenchmarkGroup()
let N=50, T=25, K=3, D=3, P=4, M_true=5, SEED=123, maxiter_bm=10, n_init_bm=10 # Use fewer inits for benchmark
    # Note: This benchmark implicitly uses Threads.@threads based on how Julia is run.
    # Compare this timing to the corresponding "Baseline" scenario.
    SUITE["RegressionHMM"]["Parallel Execution"]["N=$N T=$T K=$K D=$D P=$P n_init=$n_init_bm"] = @benchmarkable(
        run_em!(RegressionHMMParams, data; n_init=$n_init_bm, maxiter=$maxiter_bm, tol=1e-4, verbose=false),
        setup = begin
                Random.seed!($SEED)
            sim_params = SF.HMMRegressionSimulationParams(
                K = $K,
                mu_dist = Normal(0, 3.0),  # Override mu_sd
                eta_dist = Normal(0, 1.0) # Override eta_sd
            )
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T_max = $T, N = $N, D = $D, J = $M_true + 1, 
                params = sim_params
            );
                # Create ragged arrays
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, T_vec = SF.create_ragged_vector(sim.k, sim.T) 
                # --- Generate Basis Matrix Ragged Array ---
                Φ_rag = [monomial_basis(k_seq, $P-1) for k_seq in k_rag]
                # --- Create Data Struct ---
                # Use Φ_rag and P instead of k_rag and M
                data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P)
                params = initialize_params(RegressionHMMParams, $SEED + 1, data)
        end,
        evals = 1
    )
end

println("Benchmark suite initialized.")

# To run: julia --project=benchmark benchmark/benchmarks.jl 
# (or include and run manually) 

# --- Run and Save Results --- 
# Note: Adjust seconds/samples for faster/slower runs
println("Running benchmarks... (this may take a while)")
results = run(SUITE, verbose = true, seconds = 10) 

results_path = joinpath(@__DIR__, "benchmark_results.json")
BenchmarkTools.save(results_path, results)
println("Benchmark results saved to: ", results_path) 