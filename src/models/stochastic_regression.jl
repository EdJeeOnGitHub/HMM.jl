# --- Stochastic Regression HMM ---
# Defines the data structures and algorithms for Stochastic EM with Robbins-Monro
# updating for Regression Hidden Markov Models.

# Assumes inclusion in the main HMM module scope.
# Base types (AbstractHMMData) should be available from included base.jl

"""
    StochasticEMConfig

Configuration structure for Stochastic EM with Robbins-Monro updating.

# Fields
- `τ::Int`: Robbins-Monro updating parameter
- `t::Int`: Current iteration count for learning rate scheduling
- `batch_size::Int`: Size of mini-batch to use for stochastic updates
"""
mutable struct StochasticEMConfig{F}
    weight_fn::F
    t::Int
    full_batch_step::Int
    batch_size::Int
    max_data_size::Float64
end

function StochasticEMConfig(; weight_fn = x -> 1/x, t=0, full_batch_step=10, batch_size=20, max_data_size=Inf)
    return StochasticEMConfig(weight_fn, t, full_batch_step, batch_size, max_data_size)
end




