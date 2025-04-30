# --- Regression HMM ---
# Defines the data structures for Hidden Markov Models where emission parameters
# depend on observed covariates.

# Assumes inclusion in the main HMM module scope.
# Base types (AbstractHMMData) should be available from included base.jl

# export RegressionHMMData # Exports handled by main module


"""
    RegressionHMMData{T} <: AbstractHMMData

Data structure for a Regression Hidden Markov Model with an auxiliary regression target.

# Fields
- `y_rag::Vector{Vector{T}}`: Ragged array of primary observation sequences.
- `c_rag::Vector{Vector{T}}`: Ragged array of auxiliary regression target sequences.
- `x_rag::Vector{Vector{T}}`: Ragged array of 1D covariate sequences (input for polynomial regression).
- `K::Int`: Number of hidden states.
- `M::Int`: Degree of the polynomial regression (determines number of coefficients needed).
"""
struct RegressionHMMData{T} <: AbstractHMMData
    y_rag::Vector{Vector{T}}
    c_rag::Vector{Vector{T}} # Added auxiliary target
    k_rag::Vector{Vector{T}} # Simplified covariate input (1D)
    D::Int # Number of mixture components
    K::Int # Number of hidden states
    M::Int # Changed from P to M (Polynomial Degree)
end 

"""
    RegressionHMMParams{T} <: AbstractHMMParams

Parameter structure for a Regression Hidden Markov Model with mixture components 
and auxiliary regression, stored within a ComponentArray.

# Fields
- `ca::ComponentArray{T}`: Stores the model parameters accessible via keys:
    - `:η_raw`: Vector{T}, mixture component locations.
    - `:η_θ`: Vector{T}, mixture component weights.
    - `:ω`: Vector{T}, base emission means for `y`, length `K`.
    - `:T_list`: Vector{Vector{T}}, transition probability vectors.
    - `:σ`: T, standard deviation for `y` emissions.
    - `:β`: Matrix{T}, regression coefficients for `c` ~ poly(`x`), size `D x K x (M+1)`.
    - `:σ_f`: T, standard deviation for the auxiliary regression `c`.
"""
struct RegressionHMMParams{T} <: AbstractHMMParams
    ca::ComponentArray{T}
end 


function RegressionHMMParams(; η_raw::Vector{T}, η_θ::Vector{T}, ω::Vector{T}, T_list::Vector{Vector{T}}, σ::T, beta::Matrix{T}, sigma_f::T) where {T}
    ca = ComponentArray(η_raw = η_raw, η_θ = η_θ, ω=ω, T_list=T_list, σ=σ, beta=beta, sigma_f=sigma_f)
    return RegressionHMMParams{T}(ca)
end

# Allow accessing fields directly, e.g., params.ω
Base.getproperty(p::RegressionHMMParams, s::Symbol) = getproperty(getfield(p, :ca), s)
Base.setproperty!(p::RegressionHMMParams, s::Symbol, v) = setproperty!(getfield(p, :ca), s, v)
Base.propertynames(p::RegressionHMMParams) = propertynames(getfield(p, :ca))


"""
    initialize_params(::Type{RegressionHMMParams}, seed::Int, data::RegressionHMMData)

Initialize parameters for a Regression HMM with auxiliary regression.

# Arguments
- `::Type{RegressionHMMParams}`: Specifies the type of parameters to initialize.
- `seed::Int`: Random seed for reproducibility.
- `data::RegressionHMMData`: The data used to inform initialization.

# Returns
- `RegressionHMMParams`: An initialized parameter object.
"""
function initialize_params(::Type{RegressionHMMParams}, seed::Int, data::RegressionHMMData{T}) where {T}
    Random.seed!(seed)
    K = data.K
    D = data.D
    M = data.M
    
    # Get min and max of all observations
    y_min = minimum(minimum(y) for y in data.y_rag)
    y_max = maximum(maximum(y) for y in data.y_rag)
    y_range = y_max - y_min
    
    # Initialize η values evenly spaced between min and max
    η_range = range(y_min, y_max, length=D)
    η_raw = collect(η_range)
    
    # Add small random noise to break symmetry
    η_raw .+= randn(D) * 0.1 * y_range / D
    
    # Initialize ω with random values
    ω = randn(K)


    c_vec = reduce(vcat, data.c_rag)
    k_vec = reduce(vcat, data.k_rag)
    
    # Initialize beta coefficients for polynomial regression
    # add random noise to break symmetry
    beta = zeros(D, K, M+1)
    for d in 1:D, k in 1:K
        X = hcat([k_vec.^p for p in 0:M]...)
        Y = c_vec
        β = X \ Y
        beta[d, k, :] = β + randn(M+1) * 0.1
    end

    ca = ComponentArray(
    η_raw = η_raw,
        η_θ = rand(Dirichlet(ones(D) / D)),
        ω = ω,
        T_list = [rand(Dirichlet(ones(K) / K)) for _ in 1:K],
        σ = abs(randn()) + 0.1,
        beta = beta,
        sigma_f = abs(randn()) + 0.1
        )
    return RegressionHMMParams{T}(ca)
end 

"""
    logdensity(params::RegressionHMMParams, data::RegressionHMMData)

Calculate the log density (likelihood + log priors) for the Regression HMM 
with auxiliary regression, adapted from `logdensity_mixture_regression`.

# Arguments
- `params::RegressionHMMParams`: The model parameters.
- `data::RegressionHMMData`: The observed data.

# Returns
- `Float64`: The total log density.
"""
function logdensity(params::RegressionHMMParams{T}, data::RegressionHMMData{T}) where {T}
    (; y_rag, c_rag, k_rag, K, D, M) = data
    (; η_raw, η_θ, ω, T_list, σ, beta, sigma_f) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    ω_sorted = sort(ω)
    η_sorted = sort(η_raw)

    logp = 0.0

    # === Priors ===
    logp += sum(logpdf(Normal(0,1), ω))
    for i in 1:K
        logp += logpdf(Dirichlet(ones(K) / K), T_list[i])
    end
    logp += logpdf(Truncated(Normal(0,1), 0, Inf), σ)
    logp += sum(logpdf(Normal(0,1), η_raw))
    logp += logpdf(Dirichlet(ones(D) / D), η_θ)

    # Prior over regression parameters
    for d in 1:D, k in 1:K
        logp += sum(logpdf(Normal(0, 5), beta[d, k, :]))  # you can tune the prior scale
    end

    # Prior over sigma_f
    logp += logpdf(Truncated(Normal(0,1), 0, Inf), sigma_f)

    # === Likelihood ===
    N = length(y_rag)
    for i in 1:N
        y_seq = y_rag[i]
        c_seq = c_rag[i]
        k_seq = k_rag[i]

        # Precompute polynomial features
        Φ = hcat([k_seq.^p for p in 0:M]...)  # T × (M+1)

        lp_d = zeros(eltype(ω), D)
        for d in 1:D
            # Compute regression mean f_dik for all k and t
            f_dik = [Φ * beta[d, k, :] for k in 1:K]
            ω_d = ω_sorted .+ η_sorted[d]

            logα_d = forward_logalpha_f(y_seq, c_seq, π_s, T_mat, ω_d, σ, f_dik, sigma_f)
            lp_d[d] = logsumexp(logα_d[:, end])
        end
        logp += logsumexp(lp_d .+ log.(η_θ))
    end

    return logp
end


