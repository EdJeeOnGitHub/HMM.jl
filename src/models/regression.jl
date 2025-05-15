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
- `Φ_rag::Vector{Matrix{T}}`: Ragged array of basis matrices for each sequence.
- `D::Int`: Number of mixture components.
- `K::Int`: Number of hidden states.
- `P::Int`: Number of basis functions (columns in Φ_rag[i]).
"""
struct RegressionHMMData{T} <: AbstractHMMData
    y_rag::Vector{Vector{T}}
    c_rag::Vector{Vector{T}} # Added auxiliary target
    Φ_rag::Vector{Matrix{T}} # Basis matrix for each sequence
    D::Int # Number of mixture components
    K::Int # Number of hidden states
    P::Int # Number of basis functions (columns in Φ_rag[i])
end 

"""
    RegressionHMMParams{T} <: AbstractHMMParams

Parameter structure for a Regression Hidden Markov Model with mixture components 
and auxiliary regression.

# Fields
- `η_raw::Vector{T}`: Vector{T}, mixture component locations.
- `η_θ::Vector{T}`: Vector{T}, mixture component weights.
- `ω::Vector{T}`: Vector{T}, base emission means for `y`, length `K`.
- `T_list::Vector{Vector{T}}`: Vector{Vector{T}}, transition probability vectors.
- `σ::T`: T, standard deviation for `y` emissions.
- `beta::Array{T,3}`: Array{T,3}, regression coefficients for `c` ~ `Φ*β`, size `D x K x P`.
- `sigma_f::T`: T, standard deviation for the auxiliary regression `c`.
"""
mutable struct RegressionHMMParams{T} <: AbstractHMMParams
    η_raw::Vector{T}
    η_θ::Vector{T}
    ω::Vector{T}
    T_list::Vector{Vector{T}}
    σ::T
    beta::Array{T,3}  # Regression coefficients, size D x K x P
    sigma_f::T
end 


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
    P = data.P
    
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
    Φ_all = reduce(vcat, data.Φ_rag) # Concatenate all basis matrices
    
    # Initialize beta coefficients for polynomial regression
    # add random noise to break symmetry
    beta = zeros(T, D, K, P) # Ensure type T
    for d in 1:D, k in 1:K
        # Perform least squares on the concatenated data
        X = Φ_all 
        Y = c_vec 
        # Ensure X and Y have compatible dimensions before solving
        if size(X, 1) == length(Y)
            β = X \ Y
            # Ensure β has the correct length P before assignment
            if length(β) == P
                beta[d, k, :] = β .+ randn(T, P) .* T(0.1) # Add typed noise
            else
                println("Warning: Least squares solution for beta[$d, $k, :] has length $(length(β)), expected $P. Initializing with noise.")
                beta[d, k, :] = randn(T, P) .* T(0.1)
            end
        else
            println("Warning: Mismatched dimensions for least squares initialization of beta[$d, $k, :]. Size(X)=$(size(X)), Length(Y)=$(length(Y)). Initializing with noise.")
            beta[d, k, :] = randn(T, P) .* T(0.1) # Fallback: Initialize with noise
        end
    end

    # Initialize remaining parameters
    η_θ_init = rand(Dirichlet(ones(T, D) / D))
    T_list_init = [rand(Dirichlet(ones(T, K) / K)) for _ in 1:K]
    σ_init = abs(randn(T)) + T(0.1)
    sigma_f_init = abs(randn(T)) + T(0.1)

    # Construct the mutable struct directly
    return RegressionHMMParams{T}(η_raw, η_θ_init, ω, T_list_init, σ_init, beta, sigma_f_init)
end 

# Initialize by running the mixture version first, then adding the regression parameters
function initialize_params(::Type{RegressionHMMParams}, seed::Int, data::RegressionHMMData, n_tries::Int)
    mix_data = MixtureHMMData(data.y_rag, data.K, data.D)
    mix_params = initialize_params(MixtureHMMParams, seed, mix_data)
    best_mix_params = run_em!(mix_params, mix_data, maxiter = 5)
    best_mix_logp = logdensity(best_mix_params, mix_data)
    if !isfinite(best_mix_logp)
        best_mix_logp = -Inf
    end
    for i in 1:n_tries
        mix_params = initialize_params(MixtureHMMParams, seed+i, mix_data)
        mix_params = run_em!(mix_params, mix_data, maxiter = 5)
        mix_logp = logdensity(mix_params, mix_data)
        if isfinite(mix_logp)
            if mix_logp > best_mix_logp
                best_mix_params = mix_params
                best_mix_logp = mix_logp
            end
        end
    end
    best_reg_params = initialize_params(RegressionHMMParams, seed, data)
    best_reg_params.η_raw = best_mix_params.η_raw
    best_reg_params.η_θ = best_mix_params.η_θ
    best_reg_params.ω = best_mix_params.ω
    best_reg_params.T_list = best_mix_params.T_list
    best_reg_params.σ = best_mix_params.σ

    # perform E step ignoring the regression params
    # r_nd, γ_dict, ξ_dict = e_step(best_mix_params, mix_data)
    # update the regression parameters using the mixture responsibilities
    # m_step!(best_reg_params, data, r_nd, γ_dict, ξ_dict)
    best_reg_params = run_em!(best_reg_params, data, maxiter = 1)

    return best_reg_params
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
    (; y_rag, c_rag, Φ_rag, K, D, P) = data
    (; η_raw, η_θ, ω, T_list, σ, beta, sigma_f) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    ω_sorted = sort(ω)
    η_sorted = sort(η_raw)

    logp = 0.0

    # === Priors ===
    logp += sum(logpdf(Normal(0,1), ω))
    for i in 1:K
        t_contrib = logpdf(Dirichlet(ones(K) / K), T_list[i])
        if !isfinite(t_contrib)
            logp += t_contrib
        end
    end
    logp += logpdf(Truncated(Normal(0,1), 0, Inf), σ)
    logp += sum(logpdf(Normal(0,1), η_raw))
    η_θ_contrib = logpdf(Dirichlet(ones(D) / D), η_θ)
    if !isfinite(η_θ_contrib)
        logp += η_θ_contrib
    end

    # Prior over regression parameters
    for d in 1:D, k in 1:K
        logp += sum(logpdf(Normal(0, 100), beta[d, k, :]))  # you can tune the prior scale
    end

    # Prior over sigma_f
    logp += logpdf(Truncated(Normal(0,1), 0, Inf), sigma_f .+ 1e-6)

    # === Likelihood ===
    N = length(y_rag)
    for i in 1:N
        y_seq = y_rag[i]
        c_seq = c_rag[i]

        lp_d = zeros(eltype(ω), D)
        for d in 1:D
            # Compute regression mean f_dik for all k and t for the current sequence i
            current_Φ = Φ_rag[i] # Basis matrix for sequence i
            f_dik = [current_Φ * beta[d, k, :] for k in 1:K]
            ω_d = ω_sorted .+ η_sorted[d]

            logα_d = forward_logalpha_f(y_seq, c_seq, π_s, T_mat, ω_d, σ, f_dik, sigma_f)
            lp_d[d] = logsumexp(logα_d[:, end])
        end
        logp += logsumexp(lp_d .+ log.(η_θ))
    end

    return logp
end


