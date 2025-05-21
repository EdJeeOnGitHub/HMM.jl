# Paste the contents of your HmmSimFunctions.jl here
module TestUtils

include("../src/utils/helpers.jl") # Include helper functions

using Random, Distributions, LinearAlgebra

# --- Simulation Hyperparameters ---
Base.@kwdef struct HMMRegressionSimulationParams
    K::Int # Required for default A_true generation
    mu_dist::Distribution = Normal(0, 4.0)
    eta_dist::Distribution = Normal(0, 2.0)
    sigma_y_true::Float64 = 1.0
    A_true::Matrix{Float64} = hcat([runif_simplex(K) for _ in 1:K]...)'
    k_dist::Distribution = Normal(0, 1.0)
    c_noise_sd::Float64 = 0.1
    T_range::Tuple{Int, Int} = (2, 50) # Default T_max = 50, adjust if needed
end

# --- Default Auxiliary Regression Function ---

"""
    default_c_func(z::Int, k_t::Float64, params::HMMRegressionSimulationParams)

Default function to calculate the base mean of the auxiliary variable `c`
depending on the hidden state `z` and covariate `k_t`.
Replicates the logic from the original `hmm_generate_multiple_eta_reg`.
"""
function default_c_func(k, z, eta)
    # Alternating behavior for c based on the unit interval of z
    c_base_mean = ifelse(isodd(floor(Int, z)), -k, k)
    return c_base_mean + eta
end



"""
    hmm_generate_multiple_eta_reg(seed, K, T, N, D, J; mu_sd = 4.0, eta_sd = 2.0)

Generates simulated data for a Hidden Markov Model with regression features.

Refactored to use `HMMRegressionSimulationParams` struct and a customizable `c_func`.
"""
function hmm_generate_multiple_eta_reg(; # Removed type hints for backward compatibility if needed, add if desired
                                       seed, K, T_max, N, D, J,
                                       params::HMMRegressionSimulationParams = HMMRegressionSimulationParams(K=K),
                                       c_func = default_c_func
                                      )

    # ── 1. Shared HMM parameters from params struct ────────────────────────────
    Random.seed!(seed)

    A   = params.A_true # Use A from params
    pi  = stationary(A) # stationary dist.

    # Generate initial means, sort, and center using stationary distribution
    # Note: Requires create_stationary_omegas helper defined later
    μ_init = sort(rand(params.mu_dist, K))
    μ      = recentre_mu(μ_init, pi)

    η        = rand(params.eta_dist, D)       # random‑effect pool
    η_id     = rand(1:D, N)                   # id ↦ pool index
    η_ii     = η[η_id]                        # realised ηᵢ
    σ        = params.sigma_y_true            # Y observation noise SD from params

    # Generate individual horizons using T_range from params
    T_indiv  = rand(params.T_range[1]:params.T_range[2], N)
    # Ensure no T_indiv exceeds T_max (allocated array size)
    T_indiv = min.(T_indiv, T_max)

    # ── 2. Allocate containers (using T_max) ───────────────────────────────────
    Z = zeros(Int,    N, T_max)      # hidden states
    Y = zeros(Float64, N, T_max)     # observations

    # ── 3. Simulate state paths and Y observations ──────────────────────────────
    for i in 1:N
        Z[i, 1] = rand(Categorical(pi))                    # initial state
        maxT_i  = T_indiv[i] # Use individual T for simulation loop

        # State path
        for t in 2:maxT_i
            Z[i, t] = rand(Categorical(A[Z[i, t - 1], :]))
        end

        # Observations Y
        for t in 1:maxT_i
            Y[i, t] = rand(Normal(μ[Z[i, t]] + η_ii[i], σ))
        end

        # Pad remaining steps if T_indiv[i] < T_max
        if maxT_i < T_max
            Y[i, (maxT_i + 1):T_max] .= -1.0e5 # Padding value
            Z[i, (maxT_i + 1):T_max] .= -100_000 # Padding value
        end
    end

    # ── 4. Generate Covariates (k, c) using params and c_func ──────────────────
    k_data = rand(params.k_dist, N, T_max) # Exogenous k using k_dist from params
    c_data = zeros(Float64, N, T_max)

    for i in 1:N
        maxT_i = T_indiv[i]
        eta_i = η_ii[i]
        for t in 1:maxT_i
            z = Z[i, t]
            k_t = k_data[i, t]

            # Calculate base mean using the provided c_func
            c_base = c_func(k_t, z, eta_i)
            # Add random effect and noise (using c_noise_sd from params)
            c_data[i, t] = c_base + rand(Normal(0, params.c_noise_sd))
        end
        # Pad remaining steps for c_data
        if maxT_i < T_max
            c_data[i, (maxT_i + 1):T_max] .= -1.0e5 # Padding value
        end
    end

    # ── 5. Package results ──────────────────────────────────────────────────────
    return (
        J       = J,
        j_idx   = rand(1:J, N),          # study index for each i
        y       = Y,
        z       = Z,
        c       = c_data,
        k       = k_data, 
        T       = T_indiv,
        η_ii    = η_ii,
        θ  = (π = pi, A = A, μ = μ, σ = σ, 
                   η_id = η_id, η = η, params=params) # Include params used
    )
end



"""
    runif_simplex(n) = Vector{Float64}

Draw a single point uniformly from the `(n‑1)`‑simplex:
non‑negative entries that sum to 1.
"""
function runif_simplex(n::Integer)
    v = rand(n)
    v ./ sum(v)
end

function stationary(P)
    A = P^1000;
    return A[1, :];
end




# function stationary(P)
#     K, L = size(P)
#     M = I - P' + ones(K, K)
#     rhs = ones(K)
#     π = M \ rhs
#     π = max.(π, 0.0)                # non‑negativity
#     π = π / sum(π)
#     return π
# end

# """
#     create_stationary_omegas(omega_init, pi_d)

# Rescales the vector `omega_init` so that:
# 1. The element in the median position is forced to 0.
# 2. The (weighted) mean with weights `pi_d` is zero.

# Returns the rescaled vector.
# """
# function create_stationary_omegas(omega_init::AbstractVector,
#                                   pi_d::AbstractVector)

#     K        = length(omega_init)
#     @assert length(pi_d) == K "pi_d must have the same length as omega_init"

#     med_idx  = fld(K + 1, 2)          # same as (K + 1) %/% 2 in R (1‑based)

#     # 1. Remove the median element
#     keep_idx = setdiff(1:K, med_idx)  # indices except the median
#     omega_sub = omega_init[keep_idx]
#     pi_sub    = pi_d[keep_idx]

#     # 2. Compute rescaling factor so weighted mean is zero
#     rescale_factor = dot(pi_sub, omega_sub) / sum(pi_sub)

#     # 3. Shift the non‑median elements and re‑insert the median (=0)
#     new_omega_sub = omega_sub .- rescale_factor

#     omega_rc = similar(omega_init)
#     omega_rc[med_idx] = 0.0
#     omega_rc[keep_idx] = new_omega_sub

#     return omega_rc
# end




function create_ragged_vector(x, t)
    N, max_T = size(x)
    ragged_x_vector = Vector{Vector{Float64}}(undef, N)
    ragged_t_vector = Vector{Vector{Int64}}(undef, N)
    for i in 1:N
        ragged_x_vector[i] = x[i, 1:t[i]]
        ragged_t_vector[i] = collect(1:convert(Int, t[i]))
    end
    return ragged_x_vector, ragged_t_vector
end



end # module TestUtils 