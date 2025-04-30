# Paste the contents of your HmmSimFunctions.jl here
module TestUtils

using Random, Distributions, LinearAlgebra


"""
    hmm_generate_multiple_eta_reg(seed, K, T, N, D, J; mu_sd = 4.0, eta_sd = 2.0)

"""
function hmm_generate_multiple_eta_reg(;seed, K, T, N, D, J,
                                       mu_sd = 4.0,
                                       eta_sd = 2.0)

    # ── 1. Shared HMM parameters ────────────────────────────────────────────────
    Random.seed!(seed)

    A   = hcat([runif_simplex(K) for _ in 1:K]...)'        # K × K transition matrix
    pi  = stationary(A)                                     # stationary dist.
    μ   = sort(rand(Normal(0, mu_sd), K))                  # ordered state means
    μ   = create_stationary_omegas(μ, pi)                  # user‑supplied helper

    η        = rand(Normal(0, eta_sd), D)                  # random‑effect pool
    η_id     = rand(1:D, N)                                # id ↦ pool index
    η_ii     = η[η_id]                                     # realised ηᵢ
    σ        = abs(randn())                                # common σ
    T_indiv  = rand(2:T, N)                                # individual horizon

    # ── 2. Allocate containers ──────────────────────────────────────────────────
    Z = zeros(Int,    N, T)      # hidden states
    Y = zeros(Float64, N, T)     # observations

    # ── 3. Simulate panel ───────────────────────────────────────────────────────
    for i in 1:N
        Z[i, 1] = rand(Categorical(pi))                    # initial state
        maxT    = T_indiv[i]

        for t in 2:T                                       # state path
            Z[i, t] = rand(Categorical(A[Z[i, t - 1], :]))
        end

        for t in 1:T                                       # observations
            if t ≤ maxT
                Y[i, t] = rand(Normal(μ[Z[i, t]] + η_ii[i], σ))
            else                                           # padded “missing” value
                Y[i, t] = -1.0e5
                Z[i, t] = -100_000
            end
        end
    end

    # ── 4. Extra covariates (k, c) ──────────────────────────────────────────────
    k_data = randn(N, T)                                   # exogenous k
    c_data = zeros(Float64, N, T)

    for i in 1:N, t in 1:T
        z = Z[i, t]
        if     z == 1; c_data[i, t] =  k_data[i, t]
        elseif z == 2; c_data[i, t] = -k_data[i, t]
        elseif z  > 2; c_data[i, t] =  sin(k_data[i, t] * z)
        end
    end
    c_data .+= η_ii .* ones(1, T) .+ 0.1 .* randn(N, T)

    # ── 5. Package results ──────────────────────────────────────────────────────
    return (
        J       = J,
        j_idx   = rand(1:J, N),          # study index for each i
        y       = Y,
        z       = Z,
        c       = c_data,
        k_data  = k_data,
        T       = T_indiv,
        η_ii    = η_ii,
        θ       = (π = pi, A = A, μ = μ, σ = σ,
                   η_id = η_id, η = η)
    )
end



"""
    runif_simplex(n) = Vector{Float64}

Draw a single point uniformly from the `(n‑1)`‑simplex:
non‑negative entries that sum to 1.
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

"""
    create_stationary_omegas(omega_init, pi_d)

Rescales the vector `omega_init` so that:
1. The element in the median position is forced to 0.
2. The (weighted) mean with weights `pi_d` is zero.

Returns the rescaled vector.
"""
function create_stationary_omegas(omega_init::AbstractVector,
                                  pi_d::AbstractVector)

    K        = length(omega_init)
    @assert length(pi_d) == K "pi_d must have the same length as omega_init"

    med_idx  = fld(K + 1, 2)          # same as (K + 1) %/% 2 in R (1‑based)

    # 1. Remove the median element
    keep_idx = setdiff(1:K, med_idx)  # indices except the median
    omega_sub = omega_init[keep_idx]
    pi_sub    = pi_d[keep_idx]

    # 2. Compute rescaling factor so weighted mean is zero
    rescale_factor = dot(pi_sub, omega_sub) / sum(pi_sub)

    # 3. Shift the non‑median elements and re‑insert the median (=0)
    new_omega_sub = omega_sub .- rescale_factor

    omega_rc = similar(omega_init)
    omega_rc[med_idx] = 0.0
    omega_rc[keep_idx] = new_omega_sub

    return omega_rc
end




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