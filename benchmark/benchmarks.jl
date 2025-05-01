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
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T = $T, N = $N, D = $D, J = $M_true + 1, 
                mu_sd = 3.0, eta_sd = 1.0
            );
            y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
            c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
            k_rag, c = SF.create_ragged_vector(sim.k_data, sim.T) 
            k_all = reduce(vcat, k_rag)
            k_min = minimum(k_all)
            k_max = maximum(k_all)
            Φ_rag = [bernstein_basis(k_seq, $P-1, k_min, k_max) for k_seq in k_rag]
            data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P)
            # Initialize params - use a different seed for initialization itself
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
                sim = SF.hmm_generate_multiple_eta_reg(
                    seed = $SEED, K = $K, T = $T, N = $N_val, D = $D, J = $M_true + 1, 
                    mu_sd = 3.0, eta_sd = 1.0
                );
                y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
                c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
                k_rag, c = SF.create_ragged_vector(sim.k_data, sim.T) 
                k_all = reduce(vcat, k_rag)
                k_min = minimum(k_all)
                k_max = maximum(k_all)
                Φ_rag = [bernstein_basis(k_seq, $P-1, k_min, k_max) for k_seq in k_rag]
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
                # Generate sequences with varying length T_val
                sim_T = sim = SF.hmm_generate_multiple_eta_reg(
                    seed = $SEED, K = $K, T = $T_val, N = $N, D = $D, J = $M_true + 1, 
                    mu_sd = 3.0, eta_sd = 1.0
                );
                # Need to use the correct sequence length T_val for creating ragged vectors
                y_rag_T, a = SF.create_ragged_vector(sim_T.y, sim_T.T) 
                c_rag_T, b = SF.create_ragged_vector(sim_T.c, sim_T.T) 
                k_rag_T, c = SF.create_ragged_vector(sim_T.k_data, sim_T.T) 
                k_all_T = reduce(vcat, k_rag_T)
                # Bounds might change with T, recalculate
                k_min_T = isempty(k_all_T) ? 0.0 : minimum(k_all_T) 
                k_max_T = isempty(k_all_T) ? 1.0 : maximum(k_all_T)
                Φ_rag_T = [bernstein_basis(k_seq, $P-1, k_min_T, k_max_T) for k_seq in k_rag_T]
                data = RegressionHMMData(y_rag_T, c_rag_T, Φ_rag_T, $D, $K, $P)
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
                sim_K = SF.hmm_generate_multiple_eta_reg(
                    seed = $SEED, K = $K_val, T = $T, N = $N, D = $D, J = $M_true + 1, 
                    mu_sd = 3.0, eta_sd = 1.0
                );
                y_rag_K, a = SF.create_ragged_vector(sim_K.y, sim_K.T)
                c_rag_K, b = SF.create_ragged_vector(sim_K.c, sim_K.T) 
                k_rag_K, c = SF.create_ragged_vector(sim_K.k_data, sim_K.T) 
                k_all_K = reduce(vcat, k_rag_K)
                k_min_K = isempty(k_all_K) ? 0.0 : minimum(k_all_K)
                k_max_K = isempty(k_all_K) ? 1.0 : maximum(k_all_K)
                Φ_rag_K = [bernstein_basis(k_seq, $P-1, k_min_K, k_max_K) for k_seq in k_rag_K]
                # Use K_val when creating data struct
                data = RegressionHMMData(y_rag_K, c_rag_K, Φ_rag_K, $D, $K_val, $P) 
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
                sim_D = SF.hmm_generate_multiple_eta_reg(
                    seed = $SEED, K = $K, T = $T, N = $N, D = $D_val, J = $M_true + 1, 
                    mu_sd = 3.0, eta_sd = 1.0
                );
                y_rag_D, a = SF.create_ragged_vector(sim_D.y, sim_D.T)
                c_rag_D, b = SF.create_ragged_vector(sim_D.c, sim_D.T) 
                k_rag_D, c = SF.create_ragged_vector(sim_D.k_data, sim_D.T) 
                k_all_D = reduce(vcat, k_rag_D)
                k_min_D = isempty(k_all_D) ? 0.0 : minimum(k_all_D)
                k_max_D = isempty(k_all_D) ? 1.0 : maximum(k_all_D)
                Φ_rag_D = [bernstein_basis(k_seq, $P-1, k_min_D, k_max_D) for k_seq in k_rag_D]
                # Use D_val when creating data struct
                data = RegressionHMMData(y_rag_D, c_rag_D, Φ_rag_D, $D_val, $K, $P) 
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
                # Simulation data is generated independent of model P
                sim_P = SF.hmm_generate_multiple_eta_reg(
                    seed = $SEED, K = $K, T = $T, N = $N, D = $D, J = $M_true + 1, 
                    mu_sd = 3.0, eta_sd = 1.0
                );
                y_rag_P, a = SF.create_ragged_vector(sim_P.y, sim_P.T)
                c_rag_P, b = SF.create_ragged_vector(sim_P.c, sim_P.T) 
                k_rag_P, c = SF.create_ragged_vector(sim_P.k_data, sim_P.T) 
                k_all_P = reduce(vcat, k_rag_P)
                k_min_P = isempty(k_all_P) ? 0.0 : minimum(k_all_P)
                k_max_P = isempty(k_all_P) ? 1.0 : maximum(k_all_P)
                # Use P_val for bernstein basis generation and data struct
                Φ_rag_P = [bernstein_basis(k_seq, $P_val-1, k_min_P, k_max_P) for k_seq in k_rag_P]
                data = RegressionHMMData(y_rag_P, c_rag_P, Φ_rag_P, $D, $K, $P_val) 
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
            sim = SF.hmm_generate_multiple_eta_reg(
                seed = $SEED, K = $K, T = $T, N = $N, D = $D, J = $M_true + 1, 
                mu_sd = 3.0, eta_sd = 1.0
            );
            y_rag, a = SF.create_ragged_vector(sim.y, sim.T)
            c_rag, b = SF.create_ragged_vector(sim.c, sim.T) 
            k_rag, c = SF.create_ragged_vector(sim.k_data, sim.T) 
            k_all = reduce(vcat, k_rag)
            k_min = minimum(k_all)
            k_max = maximum(k_all)
            Φ_rag = [bernstein_basis(k_seq, $P-1, k_min, k_max) for k_seq in k_rag]
            # Data generation is the same as baseline
            data = RegressionHMMData(y_rag, c_rag, Φ_rag, $D, $K, $P) 
            # No need to initialize params here, run_em!(Type, ...) does it internally
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