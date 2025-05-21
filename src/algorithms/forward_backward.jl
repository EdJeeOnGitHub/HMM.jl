# Forward and Backward algorithm implementations
# Assumes inclusion in main HMM module scope.

# Dependencies (LogExpFunctions, Distributions) assumed to be loaded by main module.
# Base types (AbstractHMMData/Params) and specific types (SimpleHMMData/Params) 
# should be available from included model files.
# Helper functions (_logpdf_normal) should be available from included util files.

# export forward_logalpha, backward_logbeta # Exports handled by main module

# TODO: Adapt function signatures to use AbstractHMMData/Params for dispatch?
# Current implementation is specific to SimpleHMM with Normal emissions.

function forward_logalpha(y, π1, A, μ, σ)
    Tval = promote_type(eltype(y), eltype(π1), eltype(A), eltype(μ), eltype(σ))
    K, Tsteps = length(π1), length(y)
    
    # Handle empty sequence case
    if Tsteps == 0
        return Matrix{Tval}(undef, K, 0)
    end

    logα = Matrix{Tval}(undef, K, Tsteps)
    acc = Vector{Tval}(undef, K)

    # Use _logpdf_normal directly (should be in scope)
    for j in 1:K
        logα[j, 1] = log(π1[j]) + _logpdf_normal(y[1], μ[j], σ)
    end
    
    log_A = log.(A .+ 1e-10) # Add small epsilon for stability if A has zeros

    for t in 2:Tsteps
        for j in 1:K
            lpdf = _logpdf_normal(y[t], μ[j], σ)
            for i in 1:K
                acc[i] = logα[i, t-1] + log_A[i,j] + lpdf
            end
            logα[j, t] = logsumexp(acc)
        end
    end
    return logα
end

function backward_logbeta(y_seq, T_mat, ω, σ)
    K = size(T_mat, 1)
    T_len = length(y_seq)
    Tval = promote_type(eltype(y_seq), eltype(T_mat), eltype(ω), eltype(σ))

    # Handle empty sequence case
    if T_len == 0
        return Matrix{Tval}(undef, K, 0)
    end

    logβ = Matrix{Tval}(undef, K, T_len)
    logβ[:, end] .= 0.0 # log(1)

    log_T_mat = log.(T_mat .+ 1e-10) # Add small epsilon

    for t = T_len-1:-1:1
        # Precompute log-likelihoods for time t+1
        log_likelihoods_t_plus_1 = Vector{Tval}(undef, K)
        for k_next in 1:K
            log_likelihoods_t_plus_1[k_next] = _logpdf_normal(y_seq[t+1], ω[k_next], σ)
        end

        for k in 1:K
            acc = Vector{Tval}(undef, K)
            for k_next in 1:K
                acc[k_next] = log_T_mat[k, k_next] + log_likelihoods_t_plus_1[k_next] + logβ[k_next, t+1]
            end
            logβ[k, t] = logsumexp(acc)
        end
    end
    return logβ
end 

"""
    forward_logalpha_f(y, π1, A, μ, σ)

Run the log‑space Forward algorithm.

# Arguments
- `y  :: AbstractVector{T}`      : observations, length **T**
- `π1 :: AbstractVector{T}`      : initial state probabilities, length **K**
- `A  :: AbstractMatrix{T}`      : K×K transition matrix (rows sum to 1)
- `μ  :: AbstractVector{T}`      : emission means, length **K**
- `σ  :: AbstractVector{T}`      : emission std. devs., length **K**

`T` can be `Float64`, `ForwardDiff.Dual`, `ReverseDiff.TrackedReal`, etc.

f[K][T] 

# Returns
`logα :: Matrix{T}` of size **K × T**, where `logα[j, t] = log p(zₜ = j | y₁:ₜ)`.
"""
function forward_logalpha_f(y, c, π1, A, μ, σ, f, σ_f) 
    Tval = promote_type(eltype(y), eltype(c), eltype(π1), eltype(A), eltype(μ), eltype(σ), eltype(f), eltype(σ_f))

    K       = length(π1)
    Tsteps  = length(y)
    @assert size(A) == (K, K)
    @assert length(μ) == K 

    logα = Matrix{Tval}(undef, K, Tsteps)         # output
    acc  = Vector{Tval}(undef, K)                 # workspace


    log_A = log.(A)

    # 1) Initialisation: log p(z₁=j) + log p(y₁ | z₁=j)
    for j in 1:K
        lpdf_c = _logpdf_normal(c[1], f[j][1], σ_f)
        logα[j, 1] = log(π1[j]) + _logpdf_normal(y[1], μ[j], σ) + lpdf_c
    end


    # 2) Recursion
    for t in 2:Tsteps
        for j in 1:K
            lpdf = _logpdf_normal(y[t], μ[j], σ)   # same for all i
            lpdf_c = _logpdf_normal(c[t], f[j][t], σ_f)
            for i in 1:K
                acc[i] = logα[i, t-1] + log_A[i, j] + lpdf + lpdf_c
            end
            logα[j, t] = logsumexp(acc)               # stable ∑ in log‑space
        end
    end

    return logα
end

function backward_logbeta_f(y, c, A, μ, σ, f, σ_f)
    Tval = promote_type(eltype(y), eltype(c), eltype(A), eltype(μ), eltype(σ), eltype(f), eltype(σ_f))
    K = size(A, 1)
    T_len = length(y)
    logβ = Matrix{Tval}(undef, K, T_len)

    logβ[:, end] .= 0.0  # log(1)

    log_A = log.(A)

    for t = T_len - 1:-1:1
        for i in 1:K
            acc = Vector{Tval}(undef, K)
            for j in 1:K
                lpdf_y = _logpdf_normal(y[t+1], μ[j], σ)
                lpdf_c = _logpdf_normal(c[t+1], f[j][t+1], σ_f)
                acc[j] = log_A[i, j] + lpdf_y + lpdf_c + logβ[j, t+1]
            end
            logβ[i, t] = logsumexp(acc)
        end
    end

    return logβ
end

