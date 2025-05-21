

    function calculate_log_mixture_prob(log_alpha_array, i, T_len, log_η_θ)
        D = length(log_η_θ)
        log_numerator = zeros(D)
        for d in 1:D
            log_numerator[d] = log_η_θ[d] + logsumexp(log_alpha_array[i, d, :, T_len])
        end
        log_total = logsumexp(log_numerator)
        log_mixture_prob = log_numerator .- log_total
        return log_mixture_prob  # still on log scale
    end


    function calculate_log_nu_itjk(
        log_alpha_i, log_beta_i, t, j, k,
        ω_d_mat, log_η_θ, log_Q, y_seq, σ
    )
        D, K = size(log_alpha_i, 1), size(log_Q, 1)
        y_t = y_seq[t]
        # -- Numerator terms in log space --
        log_top_terms = similar(log_η_θ) # D-element vector
        for d in 1:D
            ω_d_k = ω_d_mat[k, d]
            log_top_terms[d] = log_alpha_i[d, j, t-1] +
                               log_Q[j, k] +
                               logpdf(Normal(ω_d_k, σ), y_t) +
                               log_beta_i[d, k, t] +
                               log_η_θ[d]
        end
        log_top = logsumexp(log_top_terms)
    
        # -- Denominator terms in log space --
        log_bot_terms = Vector{Float64}(undef, D * K * K)
        idx = 1
        for j_prime in 1:K, k_prime in 1:K, d in 1:D
            ω_d_k = ω_d_mat[k_prime, d]
            log_bot_terms[idx] = log_alpha_i[d, j_prime, t-1] +
                log_Q[j_prime, k_prime] +
                logpdf(Normal(ω_d_k, σ), y_t) +
                log_beta_i[d, k_prime, t] +
                log_η_θ[d]
            idx += 1
        end
        log_bot = logsumexp(log_bot_terms)
    
        return log_top - log_bot
    end
    

function ed_e_step(params::MixtureHMMParams, data::MixtureHMMData)
    (; y_rag, K, D) = data
    (; η_raw, η_θ, ω, T_list, σ) = params

    T_mat = hcat(T_list...)'
    π_s = stationary(T_mat)
    ω_sorted = ω
    η_sorted = η_raw
    N = length(y_rag)
    T = maximum(length.(y_rag))
    

    log_alpha_array = zeros(N, D, K, T)
    log_beta_array = zeros(N, D, K, T)



    log_mixture_probs = zeros(N, D)
    log_nu_itjk = zeros(N, T, K, K)

    ω_d_mat = ω_sorted .+ η_sorted'
    log_η_θ = log.(η_θ)
    log_Q = log.(T_mat .+ 1e-10) # Add small epsilon for stability if T_mat has zeros

    for i in 1:N
        y_seq = y_rag[i]
        T_len = length(y_seq)
        for d in 1:D
            ω_d = ω_sorted .+ η_sorted[d]
            logα_d = forward_logalpha(y_seq, π_s, T_mat, ω_d, σ)
            logβ_d = backward_logbeta(y_seq, T_mat, ω_d, σ)
            log_alpha_array[i, d, :, 1:T_len] = logα_d
            log_beta_array[i, d, :, 1:T_len] = logβ_d
        end
    end
    # Calculate mixture probabilities
    for i in 1:N
        y_seq = y_rag[i]
        T_len = length(y_seq)
        log_mixture_probs[i, :] = calculate_log_mixture_prob(log_alpha_array, i, T_len, log_η_θ)
    end


    # Calculate log_nu_itjk
    for i in 1:N, j in 1:K, k in 1:K
        log_alpha_i = log_alpha_array[i, :, :, :]
        log_beta_i = log_beta_array[i, :, :, :]
        y_seq = y_rag[i]
        T_len = length(y_seq)
        for t in 2:T
            nu_t_idx = t - 1
            if t > T_len
                log_nu_itjk[i, nu_t_idx, j, k] = -Inf
            else
                # Calculate log_nu_itjk for the current i, j, k, t
                log_nu_itjk[i, nu_t_idx, j, k] = calculate_log_nu_itjk(
                    log_alpha_i, log_beta_i, t, j, k,
                    ω_d_mat, log_η_θ, log_Q, y_seq, σ
                )
            end
        end
    end

    # calculate z_itjl
    log_z_itjd_top = zeros(N, T, K, D)
    for i in 1:N, t in 1:T, j in 1:K, d in 1:D
        T_len = length(y_rag[i])
        if t > T_len
            log_z_itjd_top[i, t, j, d] = -Inf
        else
            log_z_itjd_top[i, t, j, d] = log_alpha_array[i, d, j, t] + log_beta_array[i, d, j, t] + log_η_θ[d]
        end
    end
    log_z_itjd_bottom = logsumexp(log_z_itjd_top, dims = [3, 4])
    log_z_itjd = log_z_itjd_top .- log_z_itjd_bottom
    # -Inf - -Inf returns NaN so we set those to -Inf - this will be 0 in the log space
    log_z_itjd[isnan.(log_z_itjd)] .= -Inf

    log_u_itj = dropdims(logsumexp(log_z_itjd, dims = 4), dims = 4)
    return log_u_itj, log_nu_itjk, log_mixture_probs, log_z_itjd
end



function estimate_two_way_intercepts_with_sd(z_itjl, y_rag, M, L)
    N, T_max = size(z_itjl, 1), size(z_itjl, 2)
    P = M + L
    XTX = zeros(P, P)
    XTy = zeros(P)

    for i in 1:N
        Ti = length(y_rag[i])
        for t in 1:Ti
            y = y_rag[i][t]
            for j in 1:M, l in 1:L
                w = z_itjl[i, t, j, l]
                if w == 0.0
                    continue
                end
                x = zeros(P)
                x[j] = 1.0
                x[M + l] = 1.0
                XTX .+= w .* (x * x')
                XTy .+= w .* x .* y
            end
        end
    end



        # T = eltype(XtWX) # Get type for lambda and I
        # lambda = T(1e-6) # Small ridge penalty
        # I_P = Diagonal(ones(T, P)) # Identity matrix of size PxP

        # # Solve the regularized system
        # params.beta[d, k, :] .= (XtWX + lambda * I_P) \ XtWY  # update β with ridge
    T = eltype(XTX) # Get type for lambda and I
    lambda = T(1e-6) # Small ridge penalty
    I_P = Diagonal(ones(P, P)) # Identity matrix of size PxP

    θ = (XTX + lambda * I_P) \ XTy
    γ = θ[1:M]
    b = θ[(M+1):end]

    # Compute weighted residual sum of squares
    wrss = 0.0
    total_weight = 0.0
    for i in 1:N
        Ti = length(y_rag[i])
        for t in 1:Ti
            y = y_rag[i][t]
            for j in 1:M, l in 1:L
                w = z_itjl[i, t, j, l]
                if w == 0.0
                    continue
                end
                y_hat = γ[j] + b[l]
                wrss += w * (y - y_hat)^2
                total_weight += w
            end
        end
    end

    dof = total_weight - P
    σ = sqrt(wrss / dof)

    return γ, b, σ
end



    function m_step_mix(log_mix)
        vec(sum(exp.(log_mix), dims = 1) ./ size(log_mix, 1))
    end


    function m_step_δk(log_u)
        u = exp.(log_u)
        δ_k = sum(u[:, 1, :], dims = 1) ./ size(u, 1)
        return vec(δ_k)
    end

    function m_step_qjk(log_nu_itjk)
        # Convert from log-space to normal space safely
        nu_itjk = exp.(log_nu_itjk)
        
        # Numerator: sum over i and t
        numerator = sum(nu_itjk, dims = (1, 2))  # size (1, 1, K, K)
        
        # Denominator: sum over i, t, and k (axis 4)
        denominator = sum(numerator, dims = 4)  # size (1, 1, K, 1)
        
        # Normalize along the final axis (k)
        q = dropdims(numerator ./ denominator; dims=(1,2))  # size (K, K)
        
        return q
    end
    




    function ed_m_step!(params, data, log_u, log_nu, log_mix, log_z)
        (; y_rag, K, D) = data

        max_Q = m_step_qjk(log_nu)
        max_η_θ = m_step_mix(log_mix)
        z_itkd = exp.(log_z)
        max_ω, max_η, max_σ = estimate_two_way_intercepts_with_sd(z_itkd, y_rag, K, D)

        max_T_list = [max_Q[i, :] for i in 1:K]

        # π_s = stationary(max_Q)
        # recent_max_ω = recentre_mu(max_ω, π_s)
        
        params.η_raw = max_η
        params.η_θ = max_η_θ
        params.ω = max_ω
        params.T_list = max_T_list
        params.σ = max_σ

        return params
    end

    function ed_run_em!(params::MixtureHMMParams, data::MixtureHMMData; maxiter::Int=1000, tol::Float64=1e-4)
        old_logp = logdensity(params, data)
        for iter in 1:maxiter
            log_u, log_nu, log_mix, log_z = ed_e_step(params, data)
            params = ed_m_step!(params, data, log_u, log_nu, log_mix, log_z)
            new_logp = logdensity(params, data)
            if new_logp < old_logp
                println("Warning: log density decreased from $old_logp to $new_logp")
            end
            println("EM Iteration $iter: logp = $(round(new_logp, digits=4))")
            # Check convergence
            if iter > 1 && abs(logdensity(params, data) - old_logp) < tol
                println("Converged at iteration $iter")

                break
            end
            old_logp = new_logp
        end
        return params
    end