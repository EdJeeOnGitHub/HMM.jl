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
                # acc[i] = logα[i, t-1] + log(A[i,j]) + lpdf # Original, log(0) possible
                acc[i] = logα[i, t-1] + log_A[i,j] + lpdf
            end
            logα[j, t] = logsumexp(acc)
        end
    end
    return logα
end

# TODO: Adapt function signatures?
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
                # acc[k_next] = log.(T_mat[k, k_next]) .+ logpdf.(Normal.(ω, σ), y_seq[t+1]) .+ logβ[:, t+1] # Original
                acc[k_next] = log_T_mat[k, k_next] + log_likelihoods_t_plus_1[k_next] + logβ[k_next, t+1]
            end
            logβ[k, t] = logsumexp(acc)
        end
    end
    return logβ
end 