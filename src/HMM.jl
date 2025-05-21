module HMM

# Core Dependencies
using Random, LinearAlgebra, Distributions, StatsBase
using LogExpFunctions: logsumexp
using Base.Threads # Added for parallel EM

# Order of includes matters!
# Base types needed first
include("models/base.jl")
export AbstractHMMData, AbstractHMMParams, AbstractHMMModel

# Utilities needed by algorithms and models
include("utils/helpers.jl")
include("utils/basis.jl") # Add include for basis functions
# include("utils/distributions.jl") # Placeholder

# Core algorithms needed by models/EM steps
include("algorithms/forward_backward.jl")
# include("algorithms/parallel.jl") # Placeholder

# Specific model implementations (structs, initialization, logdensity)
# These might depend on algorithms/utils included above
include("models/simple.jl")
include("models/mixture.jl")
include("models/regression.jl") # Add include for regression model
include("models/stochastic_regression.jl")
include("models/ed-mixture.jl")

# EM algorithm steps (may depend on models and core algorithms)
include("algorithms/em.jl") 

# --- Exports --- 
# Define the public API of the package

# Base types (optional, useful for extension)
export AbstractHMMData, AbstractHMMParams # AbstractHMMModel not used yet

# Simple HMM related exports
export SimpleHMMData, SimpleHMMParams
export initialize_params # Generic initializer dispatch placeholder
export logdensity # Generic logdensity dispatch placeholder
export run_em! # Generic EM runner dispatch placeholder
export e_step, m_step! # Export E/M steps if needed for debugging/advanced use
export ed_e_step, ed_m_step!, ed_run_em!


# Add exports for Mixture, Regression, Parallel etc. when implemented
export MixtureHMMData, MixtureHMMParams
export RegressionHMMData # Export data type
export RegressionHMMParams # Export new params type
export bernstein_basis # Export basis function
export monomial_basis # Export basis function
export hermite_basis # Export basis function
# export run_mixture_regression_em!, ...
# export run_parallel_em

# Export types
export AbstractHMMData, AbstractHMMParams
export SimpleHMMData, SimpleHMMParams
export MixtureHMMData, MixtureHMMParams
export RegressionHMMData, RegressionHMMParams
export StochasticEMConfig

# Export functions
export initialize_params, logdensity
export e_step, m_step!, run_em!
export stochastic_e_step, stochastic_m_step!, run_stochastic_em!

end # module HMM



