# This file defines structures and functions for the Simple HMM
# It assumes it's included into the main HmmEM module scope.

# Dependencies like Random, Distributions, ComponentArrays, logsumexp
# should be imported in the main HmmEM.jl file.
# Base types like AbstractHMMData, AbstractHMMParams should be available
# because models/base.jl is included before this file.
# Functions like forward_logalpha, stationary should be available
# because the algorithm/util files are included before this file.


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
struct SimpleHMMParams{T} <: AbstractHMMParams
    # Using ComponentArray for now for easy access, matches plan fields
    ca::ComponentArray{T}
end

# Constructor to match the planned fields
function SimpleHMMParams(; ω::Vector{T}, T_list::Vector{Vector{T}}, σ::T) where {T}
    # Potential place to add validation
    K = length(ω)
    @assert length(T_list) == K "T_list must have K=$K elements"
    @assert all(length(row) == K for row in T_list) "Each row in T_list must have K=$K elements"
    @assert all(sum(row) ≈ 1.0 for row in T_list) "Each row in T_list must sum to 1"
    @assert σ > 0 "Standard deviation σ must be positive"

    ca = ComponentArray(ω = ω, T_list = T_list, σ = σ)
    return SimpleHMMParams{T}(ca)
end

# Allow accessing fields directly, e.g., params.ω
Base.getproperty(p::SimpleHMMParams, s::Symbol) = getproperty(getfield(p, :ca), s)
Base.setproperty!(p::SimpleHMMParams, s::Symbol, v) = setproperty!(getfield(p, :ca), s, v)
Base.propertynames(p::SimpleHMMParams) = propertynames(getfield(p, :ca))

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
    # Create ComponentArray first, then wrap in SimpleHMMParams
    ca = ComponentArray( 
        ω = randn(T, K),
        T_list = [rand(Dirichlet(ones(T, K) / K)) for _ in 1:K],
        σ = abs(randn(T)) + T(0.1) # Ensure positivity and match type T
    )
    return SimpleHMMParams{T}(ca)
end 