# This file defines structures and functions for the Simple HMM



# --- Data Structure ---
"""
    SimpleHMMData{T} <: AbstractHMMData

Data structure for a simple Hidden Markov Model with Normal emissions.

Fields:
- `y_rag::Vector{Vector{T}}`: Ragged array of observation sequences.
- `K::Int`: Number of hidden states.
"""
struct SimpleHMMData{T} <: AbstractHMMData
    y_rag::Vector{Vector{T}}
    K::Int
end

# --- Parameter Structure ---
"""
    SimpleHMMParams{T} <: AbstractHMMParams

Parameter structure for a simple Hidden Markov Model.

Fields:
- `ω::Vector{T}`: State means (will be sorted internally where needed).
- `T_list::Vector{Vector{T}}`: List of transition probability vectors (rows of transition matrix).
- `σ::T`: Standard deviation of emission distributions (shared across states).
"""
mutable struct SimpleHMMParams{T} <: AbstractHMMParams
    ω::Vector{T}
    T_list::Vector{Vector{T}}
    σ::T
end

# --- Logdensity ---
"""
    logdensity(params::SimpleHMMParams, data::SimpleHMMData)

Calculate the log-likelihood of the data given the parameters, plus log-priors.
"""
function logdensity(params::SimpleHMMParams, data::SimpleHMMData)
    (; y_rag, K) = data
    (; ω, T_list, σ) = params # Access fields via getproperty overload

    T_mat = hcat(T_list...)'
    # Assume stationary distribution is uniform for prior/likelihood calculation start
    π = ones(K) / K  # uniform prior for initial state distribution 
    ω_sorted = sort(ω) # Sorting needed if model assumes ordered states

    logp = 0.0
    # Priors
    logp += sum(logpdf(Normal(0,1), ω)) # Prior on means
    for i in 1:K
        # Prior for each row of the transition matrix
        logp += logpdf(Dirichlet(ones(K) / K), T_list[i])
    end
    logp += logpdf(Truncated(Normal(0,1), 0, Inf), σ) # Prior on std dev

    # Likelihood of sequences
    for i in 1:length(y_rag)
        # Call forward_logalpha directly (should be in main module scope)
        # Assuming forward_logalpha signature is: (y_seq, π_initial, T_matrix, means, std_dev)
        if length(y_rag[i]) > 0 # Handle empty sequences if they can occur
            logα = forward_logalpha(y_rag[i], π, T_mat, ω_sorted, σ)
            logp += logsumexp(logα[:, end])
        end
    end

    return logp
end

# --- Random Initialization ---
"""
    initialize_params(::Type{SimpleHMMParams}, seed::Int, data::SimpleHMMData)

Initialize parameters for a SimpleHMM randomly.
"""
function initialize_params(::Type{SimpleHMMParams}, seed::Int, data::SimpleHMMData{T}) where {T}
    Random.seed!(seed)
    K = data.K
    # Directly construct the mutable struct
    ω_init = randn(T, K)
    T_list_init = [rand(Dirichlet(ones(T, K) / K)) for _ in 1:K]
    σ_init = abs(randn(T)) + T(0.1) # Ensure positivity and match type T
    return SimpleHMMParams{T}(ω_init, T_list_init, σ_init)
end 