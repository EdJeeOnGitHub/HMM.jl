# HmmEM.jl

A Julia package for Hidden Markov Models with Expectation-Maximization.

*This package is under development.*

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/EdJeeOnGitHub/HMM.jl")
```

## Usage

Here are basic examples demonstrating how to use the package for different HMM types.

```julia
using HMM
using Random, Distributions, LinearAlgebra
```

### 1. Simple HMM

```julia
# --- 1. Data Setup ---
# Simulate or load your data as a vector of sequences (ragged array)
Random.seed!(123)
K_true = 3 # True number of states
N = 50    # Number of sequences
T_max = 30 # Max sequence length
y_rag = Vector{Vector{Float64}}()
for _ in 1:N
    T = rand(10:T_max) # Variable sequence lengths
    # Replace with your actual data simulation/loading
    dummy_seq = randn(T) .+ rand(1:K_true) * 2.0 
    push!(y_rag, dummy_seq)
end

# Create the data structure
K_model = 3 # Number of states to fit
simple_data = SimpleHMMData(y_rag, K_model)

# --- 2. Run EM --- 
# Use the parallel runner with multiple initializations
best_params_simple = run_em!(SimpleHMMParams, simple_data; 
                             n_init=10,    # Number of random initializations
                             maxiter=100,  # Max iterations per run
                             tol=1e-4,     # Convergence tolerance
                             verbose=true);

# --- 3. Inspect Results --- 
println("Estimated Simple HMM Parameters:")
println("  Means (ω): ", round.(best_params_simple.ω, digits=3))
println("  Std Dev (σ): ", round(best_params_simple.σ, digits=3))
println("  Transition Matrix (T):")
display(round.(hcat(best_params_simple.T_list...)', digits=3))
```

### 2. Mixture HMM

```julia
# --- 1. Data Setup ---
# Simulate or load data (similar structure to SimpleHMM)
Random.seed!(456)
K_true = 3 # True number of states
D_true = 2 # True number of mixture components
N = 80
T_max = 25
y_rag_mix = Vector{Vector{Float64}}()
for _ in 1:N
    T = rand(15:T_max)
    # Replace with your actual data simulation/loading
    component = rand(1:D_true)
    state_effect = rand(1:K_true) * 1.5
    mix_effect = (component == 1) ? -2.0 : 2.0
    dummy_seq = randn(T) .+ state_effect .+ mix_effect
    push!(y_rag_mix, dummy_seq)
end

# Create the data structure
K_model = 3
D_model = 2
mixture_data = MixtureHMMData(y_rag_mix, K_model, D_model)

# --- 2. Run EM --- 
best_params_mix = run_em!(MixtureHMMParams, mixture_data; 
                          n_init=20, 
                          maxiter=100, 
                          tol=1e-4, 
                          verbose=true);

# --- 3. Inspect Results ---
println("\nEstimated Mixture HMM Parameters:")
println("  Base Means (ω): ", round.(best_params_mix.ω, digits=3))
println("  Mixture Locs (η_raw): ", round.(best_params_mix.η_raw, digits=3))
println("  Mixture Weights (η_θ): ", round.(best_params_mix.η_θ, digits=3))
println("  Std Dev (σ): ", round(best_params_mix.σ, digits=3))
println("  Transition Matrix (T):")
display(round.(hcat(best_params_mix.T_list...)', digits=3))
```

### 3. Regression HMM (Mixture Regression Variant)

This example uses the implemented model which includes mixture components and an auxiliary regression target `c` depending on a covariate `x` (or `k`).

```julia
# --- 1. Data Setup ---
Random.seed!(789)
K_true = 2
D_true = 2
M = 1      # Polynomial degree (linear)
N = 60
T_max = 20

y_rag_reg = Vector{Vector{Float64}}()
c_rag_reg = Vector{Vector{Float64}}()
x_rag_reg = Vector{Vector{Float64}}() # Covariate 'x' (or 'k')

for _ in 1:N
    T = rand(10:T_max)
    y_seq = zeros(T)
    c_seq = zeros(T)
    x_seq = randn(T) * 2.0 # Generate covariate values
    
    # Simple simulation logic (replace with yours)
    true_state = 1
    true_comp = rand(1:D_true)
    true_beta_dk = randn(M + 1) .* [0.5, 0.3] # Example coefficients
    
    for t in 1:T
        # State transition (dummy)
        if rand() < 0.1 true_state = mod1(true_state + 1, K_true) end
        
        # Generate c based on x and state/component (dummy)
        phi_t = [x_seq[t]^p for p in 0:M]
        c_mean = dot(true_beta_dk, phi_t) + (true_comp == 1 ? -1 : 1)
        c_seq[t] = c_mean + randn() * 0.5 # True sigma_f = 0.5
        
        # Generate y based on state/component (dummy)
        y_mean = (true_state == 1 ? -1.5 : 1.5) + (true_comp == 1 ? -1 : 1)
        y_seq[t] = y_mean + randn() * 1.0 # True sigma = 1.0
    end
    
    push!(y_rag_reg, y_seq)
    push!(c_rag_reg, c_seq)
    push!(x_rag_reg, x_seq)
end

# Create the data structure
K_model = 2
D_model = 2
M_model = 1
regression_data = RegressionHMMData(y_rag_reg, c_rag_reg, x_rag_reg, K_model, D_model, M_model)

# --- 2. Run EM --- 
# Note: Regression EM can be slower and require more initializations/iterations
best_params_reg = run_em!(RegressionHMMParams, regression_data; 
                          n_init=5,  # Might need more
                          maxiter=50, # Might need more
                          tol=1e-3, 
                          verbose=true);

# --- 3. Inspect Results ---
# Access parameters via property names (due to ComponentArray wrapper)
println("\nEstimated Regression HMM Parameters:")
println("  Base Means (ω): ", round.(best_params_reg.ω, digits=3))
println("  Mixture Locs (η_raw): ", round.(best_params_reg.η_raw, digits=3))
println("  Mixture Weights (η_θ): ", round.(best_params_reg.η_θ, digits=3))
println("  Std Dev (σ for y): ", round(best_params_reg.σ, digits=3))
println("  Std Dev (σ_f for c): ", round(best_params_reg.sigma_f, digits=3))
println("  Transition Matrix (T):")
display(round.(hcat(best_params_reg.T_list...)', digits=3))
println("  Regression Coeffs (β - showing for d=1, k=1):")
display(round.(best_params_reg.beta[1, 1, :], digits=3))
```
