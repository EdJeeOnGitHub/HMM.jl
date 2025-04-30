# Defines structures for the Mixture HMM.
# Assumes inclusion in main HMM module scope.

# Dependencies (ComponentArrays) assumed loaded by main module.
# Base types (AbstractHMMData, AbstractHMMParams) assumed available.


# --- Data Structure ---
"""
    MixtureHMMData{T} <: AbstractHMMData

Data structure for a Mixture Hidden Markov Model with Normal emissions.

Fields:
- `y_rag::Vector{Vector{T}}`: Ragged array of observation sequences.
- `K::Int`: Number of hidden states.
- `D::Int`: Number of mixture components (e.g., ability groups).
"""
struct MixtureHMMData{T} <: AbstractHMMData
    y_rag::Vector{Vector{T}}
    K::Int
    D::Int
end

# --- Parameter Structure ---
"""
    MixtureHMMParams{T} <: AbstractHMMParams

Parameter structure for a Mixture Hidden Markov Model.

Fields:
- `η_raw::Vector{T}`: Raw mixture component means (location shifts).
- `η_θ::Vector{T}`: Mixture component probabilities (Dirichlet distribution).
- `ω::Vector{T}`: Base state means (will be recentered internally).
- `T_list::Vector{Vector{T}}`: List of transition probability vectors (rows of transition matrix).
- `σ::T`: Standard deviation of emission distributions (shared across states and mixtures).
"""
struct MixtureHMMParams{T} <: AbstractHMMParams
    # Using ComponentArray for easy access, mirrors plan fields
    ca::ComponentArray{T}
end

# Constructor to match the planned fields
function MixtureHMMParams(; η_raw::Vector{T}, η_θ::Vector{T}, ω::Vector{T}, T_list::Vector{Vector{T}}, σ::T) where {T}
    # Potential place to add validation
    K = length(ω)
    D = length(η_raw)
    @assert length(η_θ) == D "Length of η_θ must match D=$D"
    @assert sum(η_θ) ≈ 1.0 "Mixture probabilities η_θ must sum to 1"
    @assert all(η_θ .>= 0) "Mixture probabilities η_θ must be non-negative"
    @assert length(T_list) == K "T_list must have K=$K elements"
    @assert all(length(row) == K for row in T_list) "Each row in T_list must have K=$K elements"
    @assert all(sum(row) ≈ 1.0 for row in T_list) "Each row in T_list must sum to 1"
    @assert σ > 0 "Standard deviation σ must be positive"

    ca = ComponentArray(η_raw=η_raw, η_θ=η_θ, ω=ω, T_list=T_list, σ=σ)
    return MixtureHMMParams{T}(ca)
end

# Allow accessing fields directly, e.g., params.ω
Base.getproperty(p::MixtureHMMParams, s::Symbol) = getproperty(getfield(p, :ca), s)
Base.setproperty!(p::MixtureHMMParams, s::Symbol, v) = setproperty!(getfield(p, :ca), s, v)
Base.propertynames(p::MixtureHMMParams) = propertynames(getfield(p, :ca))

# --- Random Initialization ---
"""
    initialize_params(::Type{MixtureHMMParams}, seed::Int, data::MixtureHMMData)

Initialize parameters for a MixtureHMM randomly.

Strategy:
- η (mixture means) are initialized evenly spaced across the data range with noise.
- ω (base state means) are initialized from a standard Normal.
- T_list (transition probabilities) are initialized from a symmetric Dirichlet.
- η_θ (mixture weights) are initialized from a symmetric Dirichlet.
- σ (standard deviation) is initialized from |Normal| + 0.1.
"""
function initialize_params(::Type{MixtureHMMParams}, seed::Int, data::MixtureHMMData{T}) where {T}
    Random.seed!(seed)
    K = data.K
    D = data.D

    # Determine data range for initializing η
    # Handle empty sequences or empty data gracefully
    all_y = Iterators.flatten(y for y in data.y_rag if !isempty(y))
    y_min = isempty(all_y) ? zero(T) : minimum(all_y)
    y_max = isempty(all_y) ? zero(T) : maximum(all_y)
    y_range = max(y_max - y_min, T(0.1)) # Ensure range is positive, avoid zero range

    # Initialize η values evenly spaced, add noise
    # Use LinRange for potentially better spacing
    η_range = LinRange(y_min, y_max, D)
    η_raw = collect(T, η_range) # Ensure type T
    # Add noise proportional to spacing
    noise_scale = D > 1 ? T(0.1) * (η_range[2] - η_range[1]) : T(0.1)
    η_raw .+= randn(T, D) .* noise_scale

    # Initialize ω (base means)
    ω = randn(T, K)

    # Initialize transition matrix rows from symmetric Dirichlet
    T_list = [rand(Dirichlet(ones(T, K) ./ K)) for _ in 1:K]

    # Initialize mixture weights from symmetric Dirichlet
    η_θ = rand(Dirichlet(ones(T, D) ./ D))

    # Initialize standard deviation (ensure positivity)
    σ = abs(randn(T)) + T(0.1)

    # Create ComponentArray and wrap in the struct
    ca = ComponentArray(
        η_raw = η_raw,
        η_θ = η_θ,
        ω = ω,
        T_list = T_list,
        σ = σ
    )
    return MixtureHMMParams{T}(ca)
end

# --- Logdensity ---
"""
    logdensity(params::MixtureHMMParams, data::MixtureHMMData)

Calculate the log-likelihood of the data given the mixture HMM parameters, 
plus log-priors for the parameters.
"""
function logdensity(params::MixtureHMMParams, data::MixtureHMMData)
    (; y_rag, K, D) = data
    (; η_raw, η_θ, ω, T_list, σ) = params
    Tval = promote_type(eltype(η_raw), eltype(η_θ), eltype(ω), eltype(T_list[1]), eltype(σ))

    # Precompute transition matrix and stationary distribution
    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    
    # Sort parameters for consistent calculations (internal convention)
    ω_sorted = sort(ω)
    η_sorted = sort(η_raw)

    logp = zero(Tval)
    
    # --- Log Priors --- 
    # Prior for ω (base means)
    logp += sum(logpdf(Normal(0,1), ω))
    # Prior for transition matrix rows
    for i in 1:K
        logp += logpdf(Dirichlet(ones(Tval, K) ./ K), T_list[i])
    end
    # Prior for σ (shared standard deviation)
    logp += logpdf(Truncated(Normal(0,1), zero(Tval), Tval(Inf)), σ)
    # Prior for η_raw (mixture means/locations)
    logp += sum(logpdf(Normal(0,1), η_raw))
    # Prior for η_θ (mixture weights)
    logp += logpdf(Dirichlet(ones(Tval, D) ./ D), η_θ)

    # --- Log Likelihood --- 
    log_η_θ = log.(η_θ .+ eps(Tval)) # Precompute log weights, ensure non-zero

    for i in 1:length(y_rag)
        y_seq = y_rag[i]
        T_len = length(y_seq)
        if T_len == 0 continue end # Skip empty sequences

        # Calculate log-likelihood contribution for each mixture component d
        lp_d = Vector{Tval}(undef, D)
        for d in 1:D
            # Effective means for this component
            ω_d = ω_sorted .+ η_sorted[d]
            
            # Run forward algorithm for this component
            # Assumes forward_logalpha is in scope from main HMM module
            logα_d = forward_logalpha(y_seq, π_s, T_mat, ω_d, σ)
            
            # Log-likelihood of sequence y_seq under component d
            lp_d[d] = logsumexp(logα_d[:, end])
        end
        
        # Combine component likelihoods using mixture weights
        # log P(y_seq | params) = log Σ_d P(y_seq | component=d) * P(component=d)
        #                       = log Σ_d exp(log P(y_seq | component=d) + log P(component=d))
        logp += logsumexp(lp_d .+ log_η_θ)
    end

    return logp
end 