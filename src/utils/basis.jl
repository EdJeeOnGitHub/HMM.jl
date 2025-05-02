# Basis function utilities

"""
    bernstein_basis(x::AbstractVector{T}, degree::Int, min_val::T, max_val::T) where T<:Real -> Matrix{T}

Compute the Bernstein basis matrix for a given vector `x`.

The Bernstein basis polynomials of degree `degree` are defined on the interval `[min_val, max_val]`.

Arguments:
- `x::AbstractVector{T}`: Input vector of values.
- `degree::Int`: The degree of the Bernstein polynomial basis (P = degree + 1 basis functions).
- `min_val::T`: The minimum value of the interval for scaling `x`.
- `max_val::T`: The maximum value of the interval for scaling `x`.

Returns:
- `Matrix{T}`: A matrix where each row corresponds to an element of `x`, and each column
  corresponds to a Bernstein basis function (degree `p` from 0 to `degree`). The size is `length(x) x (degree + 1)`.
"""
function bernstein_basis(x::AbstractVector{T}, degree::Int, min_val::T, max_val::T) where T<:Real
    n = length(x)
    P = degree + 1 # Number of basis functions
    Φ = Matrix{T}(undef, n, P)
    x = collect(x)

    # Scale x to the interval [0, 1]
    x_scaled = (x .- min_val) ./ (max_val - min_val)

    # Clamp scaled values to handle potential floating point issues at boundaries
    clamp!(x_scaled, zero(T), one(T))

    for i in 1:n
        xi_scaled = x_scaled[i]
        for p in 0:degree # Iterate through basis function degrees
            coeff = binomial(degree, p)
            term1 = xi_scaled^p
            term2 = (one(T) - xi_scaled)^(degree - p)
            Φ[i, p+1] = coeff * term1 * term2 # Store in column p+1
        end
    end

    return Φ
end 

function monomial_basis(x::AbstractVector{T}, degree::Int) where T<:Real
    n = length(x)
    P = degree + 1
    Φ = Matrix{T}(undef, n, P)

    for i in 1:n
        xi = x[i]
        for p in 0:degree
            Φ[i, p+1] = xi^p
        end
    end

    return Φ
end

function hermite_basis(x::AbstractVector{T}, degree::Int) where T<:Real
    n = length(x)
    P = degree + 1
    Φ = Matrix{T}(undef, n, P)

    for i in 1:n
        xi = x[i]
        Φ[i, 1] = one(T)               # H₀(x) = 1
        if degree >= 1
            Φ[i, 2] = 2 * xi           # H₁(x) = 2x
        end
        for p in 2:degree
            Φ[i, p+1] = 2 * xi * Φ[i, p] - 2 * (p - 1) * Φ[i, p-1]
        end
    end

    return Φ
end
