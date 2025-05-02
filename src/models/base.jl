# Base types for HMM models

"""
    AbstractHMMData

Abstract supertype for all HMM data structures.
"""
abstract type AbstractHMMData end

"""
    AbstractHMMParams

Abstract supertype for all HMM parameter structures.
"""
abstract type AbstractHMMParams end

"""
    AbstractHMMModel

Abstract supertype for all HMM model representations (potentially combining data and parameters).
"""
abstract type AbstractHMMModel end

# Define common interface functions here later, e.g.:
# function n_states(model::AbstractHMMModel) end
# function n_sequences(data::AbstractHMMData) end 


function initialize_params(::Type{P}, seed::Int, data::D, n_tries::Int) where {P<:AbstractHMMParams, D<:AbstractHMMData}
    best_params = initialize_params(P, seed, data)
    best_params = run_em!(best_params, data, maxiter = 5)
    best_logp = logdensity(best_params, data)
    if !isfinite(best_logp)
        best_logp = -Inf
    end
    for i in 1:n_tries
        params = initialize_params(P, seed+i, data)
        params = run_em!(params, data, maxiter = 5)
        logp = logdensity(params, data)
        if isfinite(logp)
            if logp > best_logp
                best_params = params
                best_logp = logp
            end
        end
    end
    return best_params
end