# EM Algorithm steps (E-step, M-step) and runner.
# Assumes inclusion in main HMM module scope.

# Dependencies (LinearAlgebra, LogExpFunctions, Distributions) assumed loaded by main module.
# Base types, specific model types (SimpleHMMData/Params), logdensity function 
# should be available from included model files.
# Forward/backward algorithms should be available from included algorithm files.
# Helper functions (stationary) should be available from included util files.

# export e_step, m_step!, run_em! # Exports handled by main module

# --- E-step ---
"""
    e_step(params::SimpleHMMParams, data::SimpleHMMData)

Perform the E-step for the Simple HMM, calculating expected sufficient statistics (γ, ξ).
Returns `γ_dict`, `ξ_dict`.
"""
function e_step(params::SimpleHMMParams, data::SimpleHMMData)
    (; y_rag, K) = data
    (; ω, T_list, σ) = params # Access params via overloaded getproperty

    T_mat = hcat(T_list...)'
    
    # Calculate stationary distribution for initial state probs
    # Consider allowing user-specified initial distribution?
    π_s = stationary(T_mat) # Use helper
    
    ω_sorted = sort(ω) # Ensure means are sorted as expected by some parts

    # Type stability: Determine calculation type based on params
    Tval = promote_type(eltype(ω), eltype(σ), eltype(T_mat))
    γ_dict = Dict{Int, Matrix{Tval}}()
    ξ_dict = Dict{Int, Array{Tval,3}}()
    

    for i in 1:length(y_rag)
        y_seq = y_rag[i]
        T_len = length(y_seq)

        # Handle sequences too short for transitions
        if T_len == 0
            γ_dict[i] = Matrix{Tval}(undef, K, 0)
            ξ_dict[i] = Array{Tval, 3}(undef, K, K, 0)
            continue
        end

        # Use forward/backward functions directly (should be in scope)
        logα = forward_logalpha(y_seq, π_s, T_mat, ω_sorted, σ)
        logβ = backward_logbeta(y_seq, T_mat, ω_sorted, σ)

        # Calculate log P(y) for normalization
        log_p_y = logsumexp(logα[:, end])

        # --- Calculate γ --- 
        logγ = logα .+ logβ .- log_p_y # Normalize
        γ = exp.(logγ)
        # Numerical stability check/fix for γ
        γ ./= sum(γ, dims=1) # Ensure columns sum to 1
        γ_dict[i] = γ
        
        # --- Calculate ξ --- 
        if T_len <= 1
            # No transitions possible
             ξ_dict[i] = Array{Tval, 3}(undef, K, K, 0)
             continue
        end

        ξ = zeros(Tval, K, K, T_len-1)
        log_T_mat = log.(T_mat .+ 1e-10) # Stable log transition matrix

        for t in 1:(T_len-1)
            # Precompute log emission probabilities for t+1
            log_b_k_tplus1 = [_logpdf_normal(y_seq[t+1], ω_sorted[k], σ) for k in 1:K]
            
            # Calculate log ξ[j,k,t] using precomputed values
            log_ξ_t = [logα[j,t] + log_T_mat[j,k] + log_b_k_tplus1[k] + logβ[k,t+1] for j in 1:K, k in 1:K]
            
            # Normalize ξ for timestep t
            log_norm_const = logsumexp(log_ξ_t)
            ξ[:,:,t] .= exp.(log_ξ_t .- log_norm_const)
        end
        
        # Numerical stability check/fix for ξ
        for t in 1:size(ξ, 3)
             ξ[:,:,t] ./= sum(ξ[:,:,t]) # Ensure each time slice sums to 1
        end
        ξ_dict[i] = ξ
    end

    return γ_dict, ξ_dict
end

# --- E-step for Mixture HMM ---
"""
    e_step(params::MixtureHMMParams, data::MixtureHMMData)

Perform the E-step for the Mixture HMM.

Calculates:
- Responsibilities `r_nd[n, d]`: Probability sequence `n` belongs to mixture component `d`.
- Expected state occupancies `γ_dict[n][k, t]`: Prob. state `k` at time `t` for sequence `n` (marginalized over mixture components).
- Expected state transitions `ξ_dict[n][j, k, t]`: Prob. transitioning from state `j` to `k` between time `t` and `t+1` for sequence `n` (marginalized over mixture components).

Returns `r_nd`, `γ_dict`, `ξ_dict`.
"""
function e_step(params::MixtureHMMParams, data::MixtureHMMData)
    (; y_rag, K, D) = data
    (; η_raw, η_θ, ω, T_list, σ) = params
    
    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    ω_sorted = sort(ω)
    η_sorted = sort(η_raw)
    N = length(y_rag)
    responsibilities = zeros(N, D)
    γ_dict = Dict{Int, Matrix{Float64}}()
    ξ_dict = Dict{Int, Array{Float64,3}}()

    # Outputs
    for i in 1:N
        y_seq = y_rag[i]
        T_len = length(y_seq)
        
        # Calculate responsibilities
        lp_d = zeros(D)
        for d in 1:D
            ω_d = ω_sorted .+ η_sorted[d]
            logα_d = forward_logalpha(y_seq, π_s, T_mat, ω_d, σ)
            lp_d[d] = logsumexp(logα_d[:, end])
        end
        unnorm = lp_d .+ log.(η_θ)
        responsibilities[i, :] .= exp.(unnorm .- logsumexp(unnorm))

        # Calculate γ and ξ for each mixture component
        γ_total = zeros(K, T_len)
        ξ_total = zeros(K, K, T_len-1)

        log_αs = zeros(D,K, T_len)
        log_βs = zeros(D,K, T_len)
        
        for d in 1:D
            ω_d = ω_sorted .+ η_sorted[d]
            logα = forward_logalpha(y_seq, π_s, T_mat, ω_d, σ)
            logβ = backward_logbeta(y_seq, T_mat, ω_d, σ)

            log_αs[d,:,:] .= logα
            log_βs[d,:,:] .= logβ

            
            # Calculate γ
            logγ = logα + logβ
            for t in 1:T_len
                logγ[:, t] .-= logsumexp(logγ[:, t])
            end
            γ_total .+= responsibilities[i,d] .* exp.(logγ)
            
            # Calculate ξ
            for t in 1:(T_len-1)
                log_ξ = zeros(K, K)
                for j in 1:K, k in 1:K
                    log_ξ[j,k] = logα[j,t] + log(T_mat[j,k]) +
                                 logpdf(Normal(ω_d[k], σ), y_seq[t+1]) +
                                 logβ[k,t+1]
                end
                log_norm = logsumexp(vec(log_ξ))
                ξ_total[:,:,t] .+= responsibilities[i,d] .* exp.(log_ξ .- log_norm)
            end

        end
        
        # Normalize ξ
        for t in 1:(T_len-1)
            ξ_total[:,:,t] ./= sum(ξ_total[:,:,t])
        end
        
        γ_dict[i] = γ_total
        ξ_dict[i] = ξ_total

        if any(isnan.(γ_total)) || any(isnan.(ξ_total))
            @warn "NaNs detected at sequence $i"
           println("y_seq: $y_seq")
           println("log_αs: $log_αs")
           println("log_βs: $log_βs")
           println("γ_total: $γ_total")
           println("ξ_total: $ξ_total")
        end



    end
    return responsibilities, γ_dict, ξ_dict
end

# --- M-step for Simple HMM ---
"""
    m_step!(params::SimpleHMMParams, data::SimpleHMMData, γ_dict, ξ_dict)

Perform the M-step for the Simple HMM, updating parameters in-place.
"""
function m_step!(params::SimpleHMMParams, data::SimpleHMMData, γ_dict, ξ_dict)
    (; y_rag, K) = data
    Tval = eltype(params.ω) # Get the type
    ϵ = Tval(1e-8) # Small value for numerical stability

    # --- Update Transition Matrix --- 
    # Sum ξ over time dimension for each sequence, then sum over sequences
    expected_transitions = zeros(Tval, K, K)
    for i in 1:length(y_rag)
        ξ = ξ_dict[i]
        if size(ξ, 3) > 0 # Only if transitions exist
            expected_transitions .+= sum(ξ, dims=3)[:,:,1]
        end
    end

    # Sum expected transitions out of each state (denominator for row normalization)
    expected_departures = sum(expected_transitions, dims=2)[:,1] # Sum over k for each j

    for k in 1:K
        if expected_departures[k] > ϵ # Avoid division by zero
             params.T_list[k] .= expected_transitions[k,:] ./ expected_departures[k]
        else
             # State k is never visited or is absorbing - keep old probs or set uniform?
             # Setting uniform might be safer than keeping potentially poor initial values
             params.T_list[k] .= ones(Tval, K) ./ K
        end
    end
    # Ensure rows sum to 1 after potential numerical issues
    params.T_list .= [row ./ sum(row) for row in params.T_list]

    # --- Update Emission Parameters (ω) --- 
    numer = zeros(Tval, K)
    denom = zeros(Tval, K)
    for i in 1:length(y_rag)
        γ = γ_dict[i]
        y_seq = y_rag[i]
        for t in 1:length(y_seq)
            numer .+= γ[:,t] .* y_seq[t]
            denom .+= γ[:,t] # Expected number of times in state k
        end
    end

    ω_new = numer ./ (denom .+ ϵ)
    params.ω .= ω_new 

    # --- Update Emission Parameters (σ) --- 
    # M-step for σ (Maximum Likelihood Estimate)
    mse_sum = zero(Tval)
    total_weight = zero(Tval)
    ω_current = params.ω # Use the *just updated* ω
    for i in 1:length(y_rag)
        γ = γ_dict[i]
        y_seq = y_rag[i]
        for t in 1:length(y_seq)
            for k in 1:K
                err = y_seq[t] - ω_current[k]
                mse_sum += γ[k,t] * err^2
                total_weight += γ[k,t]
            end
        end
    end
    # Update σ if total_weight is significant
    if total_weight > ϵ
        params.σ = sqrt(mse_sum / total_weight)
    else
        # Handle case where no observations were assigned weights (e.g., all sequences empty)
        # Keep old σ or set to default? Keeping old seems safer.
    end
    # Ensure sigma doesn't collapse to zero
    params.σ = max(params.σ, Tval(1e-6))

    return params # Return modified params
end

# --- M-step for Mixture HMM ---
"""
    m_step!(params::MixtureHMMParams, data::MixtureHMMData, responsibilities, γ_dict, ξ_dict)

Perform the M-step for the Mixture HMM, updating parameters in-place based on 
responsibilities (`responsibilities`) and expected sufficient statistics (`γ_dict`, `ξ_dict`).

Updates: η_θ, T_list, ω, η_raw, σ.
Requires `recentre_mu` helper function.
"""
function m_step!(params::MixtureHMMParams, data::MixtureHMMData, responsibilities, γ_dict, ξ_dict)
    (; y_rag, K, D) = data
    (; η_raw, η_θ, ω, T_list, σ) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    N = length(y_rag)

    # Update η_θ (mixture weights)
    params.η_θ .= vec(sum(responsibilities, dims=1)) ./ N

    # --- Update Transition matrix ---
    T_counts = zeros(K, K)
    for i in 1:N
        ξ = ξ_dict[i]
        for t in 1:size(ξ, 3)
            T_counts .+= ξ[:,:,t]
        end
    end
    for k in 1:K
        params.T_list[k] .= T_counts[k,:] ./ sum(T_counts[k,:])
    end

    # --- Update ω ---
    ω_new = zeros(K)
    ω_weight = zeros(K)

    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        T_len = length(y_seq)

        for t in 1:T_len
            for d in 1:D
                r = responsibilities[i,d]
                for k in 1:K
                    ω_new[k] += r * γ[k,t] * (y_seq[t] - η_raw[d])
                    ω_weight[k] += r * γ[k,t]
                end
            end
        end
    end

    # ω_recentred = recentre_mu(ω_new, π_s)
    # params.ω .= ω_recentred
    params.ω .= ω_new ./ (ω_weight .+ 1e-8)
    params.ω .= recentre_mu(params.ω, π_s)


    # --- Update η ---
    η_new = zeros(D)
    η_weight = zeros(D)

    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        T_len = length(y_seq)

        for t in 1:T_len
            for d in 1:D
                r = responsibilities[i,d]
                for k in 1:K
                    η_new[d] += r * γ[k,t] * (y_seq[t] - params.ω[k])  # now ω is updated
                    η_weight[d] += r * γ[k,t]
                end
            end
        end
    end

    params.η_raw .= η_new ./ (η_weight .+ 1e-8)

    # --- Update σ ---
    σ_new = 0.0
    σ_weight = 0.0

    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        T_len = length(y_seq)

        for t in 1:T_len
            for d in 1:D
                r = responsibilities[i,d]
                for k in 1:K
                    err = y_seq[t] - (params.ω[k] + params.η_raw[d])
                    σ_new += r * γ[k,t] * err^2
                    σ_weight += r * γ[k,t]
                end
            end
        end
    end

    params.σ = sqrt((σ_new + 1e-6) / (σ_weight + 1e-8))

    return params
end

# --- Run EM ---
"""
    run_em!(params::SimpleHMMParams, data::SimpleHMMData; maxiter=50, tol=1e-4, verbose=false)

Run the Expectation-Maximization algorithm for a Simple HMM.
"""
function run_em!(params::SimpleHMMParams, data::SimpleHMMData; maxiter=50, tol=1e-4, verbose=false)
    # Use logdensity function directly (should be in scope)
    # Note: logdensity includes priors. For pure ML EM, might need a likelihood func.
    try 
        old_logp = logdensity(params, data)
        if !isfinite(old_logp)
            verbose && println("Initial parameters yield non-finite log density ($old_logp). Cannot start EM.")
            return params # Return initial params
        end

        for iter in 1:maxiter
            # Use E/M steps directly (should be in scope)
            γ_dict, ξ_dict = e_step(params, data)
            m_step!(params, data, γ_dict, ξ_dict)
            
            new_logp = logdensity(params, data)

            if !isfinite(new_logp)
                verbose && println("EM Iteration $iter: Log density became non-finite ($new_logp). Stopping.")
                # TODO: Optionally revert to params from previous step?
                break 
            end

            if verbose
                println("EM Iteration $iter: logp = $(round(new_logp, digits=4))")
            end

            # Check for convergence
            if abs(new_logp - old_logp) < tol
                verbose && println("Converged at iteration $iter.")
                break
            end
            old_logp = new_logp

            if iter == maxiter
                verbose && println("Reached maximum iterations ($maxiter).")
            end
        end
    catch e
        println("Error during EM: $e")
        # Potentially rethrow or handle error appropriately
        rethrow(e)
    end
    return params
end

"""
    run_em!(params::MixtureHMMParams, data::MixtureHMMData; maxiter=50, tol=1e-4, verbose=false)

Run the Expectation-Maximization algorithm for a Mixture HMM.
"""
function run_em!(params::MixtureHMMParams, data::MixtureHMMData; maxiter=50, tol=1e-4, verbose=false)
    try 
        old_logp = logdensity(params, data)
        if !isfinite(old_logp)
            verbose && println("Initial parameters yield non-finite log density ($old_logp). Cannot start EM.")
            return params # Return initial params
        end

        for iter in 1:maxiter
            # Mixture E-step returns responsibilities as the first argument
            r_nd, γ_dict, ξ_dict = e_step(params, data)
            
            # Mixture M-step requires responsibilities
            m_step!(params, data, r_nd, γ_dict, ξ_dict)
            
            # Logdensity should dispatch correctly based on types
            new_logp = logdensity(params, data)

            if !isfinite(new_logp)
                verbose && println("EM Iteration $iter: Log density became non-finite ($new_logp). Stopping.")
                break 
            end

            if verbose
                println("EM Iteration $iter (Mixture): logp = $(round(new_logp, digits=4))")
            end

            # Check for convergence
            if abs(new_logp - old_logp) < tol
                verbose && println("Converged at iteration $iter.")
                break
            end
            old_logp = new_logp

            if iter == maxiter
                verbose && println("Reached maximum iterations ($maxiter).")
            end
        end
    catch e
        println("Error during EM (Mixture): $e")
        rethrow(e)
    end
    return params
end

# --- Parallel EM Runner ---
# Need to import Base.Threads in the main HMM module

"""
    run_em!(::Type{P}, data::D; n_init=50, maxiter=100, tol=1e-4, verbose=false) where {P<:AbstractHMMParams, D<:AbstractHMMData}

Run the Expectation-Maximization algorithm with multiple random initializations in parallel.

Selects the best result based on the final log-density (likelihood + priors).

# Arguments
- `::Type{P}`: The desired parameter type (e.g., `SimpleHMMParams`, `MixtureHMMParams`). Determines which `initialize_params` method is called.
- `data::D`: The HMM data object.
- `n_init::Int=50`: Number of random initializations to run.
- `maxiter::Int=100`: Maximum EM iterations for each run.
- `tol::Float64=1e-4`: Convergence tolerance for each run.
- `verbose::Bool=true`: Whether to print progress for the overall parallel run (individual runs are silent).

# Returns
- `best_params::P`: The parameter set corresponding to the highest log-density found.
"""
function run_em!(::Type{P}, data::D; 
                 n_init=50, 
                 maxiter=100, 
                 tol=1e-4, 
                 verbose=false, # Default verbose to true for parallel runner
                 n_init_tries=10
                 ) where {P<:AbstractHMMParams, D<:AbstractHMMData}
    
    # Infer type T for storing results (assuming data.y_rag exists and is not empty)
    # Need a robust way to get T if y_rag can be empty or have empty sequences
    first_el_type = Float64 # Default type
    if !isempty(data.y_rag) && !isempty(data.y_rag[1])
       first_el_type = eltype(data.y_rag[1][1])
    end
    Tval = first_el_type 

    # Store results: Vector of tuples (logp, params) or Nothing
    results = Vector{Union{Nothing, Tuple{Tval, P}}}(undef, n_init)
    # Store logps separately for faster argmax
    logps = fill(Tval(-Inf), n_init)
    n_threads = Threads.nthreads()
    
    verbose && println("Starting $n_init EM initializations using $n_threads threads...")

    Threads.@threads for i in 1:n_init
        local_params = initialize_params(P, i, data, n_init_tries)
        try 
            run_em!(local_params, data; maxiter=maxiter, tol=tol, verbose=verbose)
            logp = logdensity(local_params, data)
            if isfinite(logp)
                logps[i] = logp
                results[i] = (logp, local_params) 
            else
                verbose && println("Run $i finished but produced non-finite log density ($logp).")
                results[i] = nothing
            end
        catch e
            verbose && println("EM run $i failed with error: $e")
            results[i] = nothing
        end
    end
    
    # Find successful runs (those with finite logp)
    successful_indices = findall(isfinite.(logps))
    
    if isempty(successful_indices)
        error("All EM initializations failed or produced non-finite log densities.")
    end

    num_successful = length(successful_indices)
    verbose && println("Completed $num_successful out of $n_init runs successfully.")
    
    # Find best result among successful runs
    best_idx_in_successful = argmax(logps[successful_indices])
    best_global_idx = successful_indices[best_idx_in_successful]
    best_logp, best_params = results[best_global_idx]
    
    if verbose
        println("Best log density found: $(round(best_logp, digits=4)) (Run $best_global_idx)")
    end
    
    return best_params, results
end 


# --- Regression HMM EM ---

function e_step(params::RegressionHMMParams, data::RegressionHMMData)
    # Unpack data fields (using Φ_rag and P now)
    (; y_rag, c_rag, Φ_rag, K, D, P) = data
    (; η_raw, η_θ, ω, T_list, σ, beta, sigma_f) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    ω_sorted = sort(ω)
    η_sorted = sort(η_raw)

    N = length(y_rag)
    responsibilities = zeros(N, D)
    γ_dict = Dict{Int, Matrix{Float64}}()
    ξ_dict = Dict{Int, Array{Float64,3}}()

    for i in 1:N
        y_seq = y_rag[i]
        T_len = length(y_seq)
        c_seq = c_rag[i]
        Φ_seq = Φ_rag[i] # Get the basis matrix for this sequence

        # Basic check for sequence length consistency
        @assert length(c_seq) == T_len "Length mismatch between y_seq and c_seq for sequence $i"
        @assert size(Φ_seq, 1) == T_len "Row count mismatch between Φ_seq and y_seq for sequence $i"

        if T_len == 0 continue end # Skip empty sequences
        
        # Calculate responsibilities
        lp_d = zeros(D)
        for d in 1:D
            # Compute regression mean using the basis matrix for this sequence
            # f_dik[k] will be a vector of length T_len
            f_dik = [Φ_seq * beta[d, k, :] for k in 1:K]
            ω_d = ω_sorted .+ η_sorted[d]
            logα_d = forward_logalpha_f(y_seq, c_seq, π_s, T_mat, ω_d, σ, f_dik, sigma_f)
            lp_d[d] = logsumexp(logα_d[:, end])
        end
        unnorm = lp_d .+ log.(η_θ)
        responsibilities[i, :] .= exp.(unnorm .- logsumexp(unnorm))

        # Calculate γ and ξ for each mixture component
        γ_total = zeros(K, T_len)
        ξ_total = zeros(K, K, T_len-1)
        
        for d in 1:D
            # Compute regression mean using the basis matrix for this sequence
            f_dik = [Φ_seq * beta[d, k, :] for k in 1:K] # Recompute or pass from above?
            ω_d = ω_sorted .+ η_sorted[d]
            logα = forward_logalpha_f(y_seq, c_seq, π_s, T_mat, ω_d, σ, f_dik, sigma_f)
            logβ = backward_logbeta_f(y_seq, c_seq, T_mat, ω_d, σ, f_dik, sigma_f)
            
            # Calculate γ
            logγ = logα + logβ
            for t in 1:T_len
                logγ[:, t] .-= logsumexp(logγ[:, t])
            end
            γ_total .+= responsibilities[i,d] .* exp.(logγ)
            
            # Calculate ξ
            # for t in 1:(T_len-1)
            #     for j in 1:K, k in 1:K
            #         ξ_total[j,k,t] += responsibilities[i,d] * exp(
            #             logα[j,t] + log(T_mat[j,k]) +
            #             logpdf(Normal(ω_d[k], σ), y_seq[t+1]) +
            #             logβ[k,t+1]
            #         )
            #     end
            # end
            for t in 1:(T_len-1)
                log_ξ = zeros(K, K)
                for j in 1:K, k in 1:K
                    log_ξ[j,k] = logα[j,t] + log(T_mat[j,k]) +
                                 logpdf(Normal(ω_d[k], σ), y_seq[t+1]) +
                                 logβ[k,t+1]
                end
                log_norm = logsumexp(vec(log_ξ))
                ξ_total[:,:,t] .+= responsibilities[i,d] .* exp.(log_ξ .- log_norm)
            end
        end
        
        # Normalize ξ
        for t in 1:(T_len-1)
            ξ_total[:,:,t] ./= sum(ξ_total[:,:,t])
        end
        
        γ_dict[i] = γ_total
        ξ_dict[i] = ξ_total
    end

    return responsibilities, γ_dict, ξ_dict
end


function m_step!(
    params::RegressionHMMParams,
    data::RegressionHMMData,
    responsibilities,
    γ_dict,
    ξ_dict
)
    (; y_rag, c_rag, Φ_rag, K, D, P) = data
    (; η_raw, η_θ, ω, T_list, σ, beta, sigma_f) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    N = length(y_rag)

    # === Update η_θ ===
    params.η_θ .= vec(sum(responsibilities, dims=1)) ./ N

    # === Update transition matrix ===
    T_counts = zeros(K, K)
    for i in 1:N
        ξ = ξ_dict[i]
        for t in 1:size(ξ, 3)
            T_counts .+= ξ[:,:,t]
        end
    end
    for k in 1:K
        params.T_list[k] .= T_counts[k,:] ./ sum(T_counts[k,:])
    end

    # === Update ω and η ===
    ω_new = zeros(K)
    ω_weight = zeros(K)

    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        for t in 1:length(y_seq)
            for d in 1:D, k in 1:K
                r = responsibilities[i,d]
                ω_new[k] += r * γ[k,t] * (y_seq[t] - η_raw[d])
                ω_weight[k] += r * γ[k,t]
            end
        end
    end
    params.ω .= ω_new ./ (ω_weight .+ 1e-8)
    params.ω .= recentre_mu(params.ω, π_s)

    η_new = zeros(D)
    η_weight = zeros(D)
    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        for t in 1:length(y_seq)
            for d in 1:D, k in 1:K
                r = responsibilities[i,d]
                η_new[d] += r * γ[k,t] * (y_seq[t] - params.ω[k])
                η_weight[d] += r * γ[k,t]
            end
        end
    end
    params.η_raw .= η_new ./ (η_weight .+ 1e-8)

    # === Update σ ===
    σ_numer = 0.0
    σ_denom = 0.0
    for i in 1:N
        y_seq = y_rag[i]
        γ = γ_dict[i]
        for t in 1:length(y_seq)
            for d in 1:D, k in 1:K
                r = responsibilities[i,d]
                err = y_seq[t] - (params.ω[k] + params.η_raw[d])
                σ_numer += r * γ[k,t] * err^2
                σ_denom += r * γ[k,t]
            end
        end
    end
    params.σ = sqrt((σ_numer + 1e-6) / (σ_denom + 1e-8))

    # === Update β and σ_f ===
    σf_numer = 0.0
    σf_denom = 0.0
    for d in 1:D, k in 1:K
        X = Float64[]
        Y = Float64[]
        W = Float64[]

        for i in 1:N
            γ = γ_dict[i]
            r = responsibilities[i,d]
            c = c_rag[i]
            Φ = Φ_rag[i]

            for t in 1:length(c)
                w = r * γ[k, t]
                if w > 1e-12  # skip near-zero weights
                    push!(W, w)
                    push!(Y, c[t])
                    push!(X, Φ[t,:]...)  # row-wise flatten
                end
            end
        end

        # Reshape into matrices
        T_obs = length(W)
        Xmat = reshape(X, P, T_obs)'  # T × P
        Wvec = Diagonal(W)
        Yvec = Y

        XtWX = Xmat' * Wvec * Xmat
        XtWY = Xmat' * Wvec * Yvec

        # Add ridge penalty for stability
        T = eltype(XtWX) # Get type for lambda and I
        lambda = T(1e-6) # Small ridge penalty
        I_P = Diagonal(ones(T, P)) # Identity matrix of size PxP

        # Solve the regularized system
        params.beta[d, k, :] .= (XtWX + lambda * I_P) \ XtWY  # update β with ridge

        residuals = Yvec .- Xmat * params.beta[d, k, :]
        σf_numer += sum(W .* residuals.^2)
        σf_denom += sum(W)
    end

    params.sigma_f = sqrt((σf_numer + 1e-6) / (σf_denom + 1e-8))

    return params
end



"""
    run_em!(params::RegressionHMMParams, data::RegressionHMMData; maxiter=50, tol=1e-4, verbose=false)

Run the Expectation-Maximization algorithm for a Regression HMM.
"""
function run_em!(params::RegressionHMMParams, data::RegressionHMMData; maxiter=50, tol=1e-4, verbose=false)
    try 
        old_logp = logdensity(params, data)
        if !isfinite(old_logp)
            verbose && println("Initial parameters yield non-finite log density ($old_logp). Cannot start EM.")
            return params # Return initial params
        end

        for iter in 1:maxiter
            # Regression HMM E-step returns responsibilities as the first argument
            r_nd, γ_dict, ξ_dict = e_step(params, data)
            
            # Regression HMM M-step requires responsibilities
            m_step!(params, data, r_nd, γ_dict, ξ_dict)
            
            # Logdensity should dispatch correctly based on types
            new_logp = logdensity(params, data)

            if !isfinite(new_logp)
                verbose && println("EM Iteration $iter: Log density became non-finite ($new_logp). Stopping.")
                break 
            end

            if verbose
                println("EM Iteration $iter (Regression): logp = $(round(new_logp, digits=4))")
            end

            # Check for convergence
            if abs(new_logp - old_logp) < tol
                verbose && println("Converged at iteration $iter.")
                break
            end
            old_logp = new_logp

            if iter == maxiter
                verbose && println("Reached maximum iterations ($maxiter).")
            end
        end
    catch e
        println("Error during EM (Regression HMM): $e")
        rethrow(e)
    end
    return params
end

