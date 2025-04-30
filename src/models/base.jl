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