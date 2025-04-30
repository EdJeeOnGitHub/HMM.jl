# Utility functions, assuming inclusion in main HMM module scope.

# Dependencies (LinearAlgebra, LogExpFunctions) assumed to be loaded by main module.

# export stationary, _logpdf_normal, recentre_mu # Exports handled by main module

function stationary(P)
    # Simple power method to find stationary distribution
    # Assumes P is ergodic and square
    K = size(P, 1)
    @assert size(P, 2) == K "Transition matrix P must be square"
    # Add check for row sums approx 1?
    try 
        A = P^1000;
        # Ensure return type matches input potentially
        Tval = eltype(P)
        stat_dist = Tval.(A[1, :])
        # Normalize for safety
        return stat_dist ./ sum(stat_dist)
    catch e
        println("Error calculating stationary distribution (matrix power method): $e")
        # Fallback or rethrow? Fallback to uniform for now.
        Tval = eltype(P)
        return ones(Tval, K) ./ K
    end
end

@inline function _logpdf_normal(y, μ, σ)
    T = promote_type(typeof(y), typeof(μ), typeof(σ))
    # Manual logpdf for Normal distribution for potential performance/AD compatibility
    # Ensure sigma is positive
    σ_safe = max(σ, eps(typeof(σ)))
    return -T(0.5) * (log(T(2π)) + 2 * log(σ_safe) + ((y - μ)/σ_safe)^2)
end

"""
    recentre_mu(mu_raw, π1)

Fix the median element of `mu_raw` at zero **and** rescale the remaining
entries so that the π-weighted mean of the whole vector is zero.

Ensures identifiability for mixture models.

# Arguments
- `mu_raw :: AbstractVector{<:Real}` : raw state means.
- `π1     :: AbstractVector{<:Real}` : stationary distribution (positive, sums to 1).

# Returns
A new vector `mu` with the same element type as the inputs.
"""
function recentre_mu(mu_raw::AbstractVector{T}, π1::AbstractVector{T}) where T

    K = length(mu_raw)
    @assert length(π1) == K "mu_raw and π1 must have equal length"
    @assert K > 0 "Input vectors cannot be empty"
    @assert sum(π1) ≈ 1.0 "π1 must sum to 1"
    @assert all(π1 .>= 0) "π1 must be non-negative"

    # Handle edge case K=1
    if K == 1
        return zeros(T, 1)
    end

    median_idx  = (K + 1) ÷ 2                      # 1-based median index
    mu          = similar(mu_raw)                  # allocate output
    
    mu[median_idx] = zero(T)                       # 1. pin median to 0

    # 2. Collect the non-median elements (views avoid extra copies)
    @views begin
        inds = vcat(1:median_idx-1, median_idx+1:K)  # indices excluding median
        μsub = mu_raw[inds]
        πsub = π1[inds]

        # 3. Rescale so the π-weighted mean is zero
        sum_πsub = sum(πsub)
        rescale = if sum_πsub > eps(T)
            dot(πsub, μsub) / sum_πsub
        else
            zero(T) # Avoid division by zero if weights are zero
        end
        mu[inds] = μsub .- rescale                  # 4. write back
    end
    return mu
end 