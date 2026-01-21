####
#### Jacobian Computation Utilities for DEER Algorithm
####
#### Provides utilities for computing Jacobians and their approximations:
#### - Full Jacobian computation
#### - Diagonal extraction
#### - Stochastic diagonal estimation (Hutchinson's method)
#### - JVP (Jacobian-vector product) wrappers
####

using Random: AbstractRNG, default_rng

####
#### Full Jacobian Computation
####

"""
    jacobian(f, x)

Compute the full Jacobian matrix of `f` at `x`.

Returns a matrix J where J[i,j] = ∂f_i/∂x_j.

This is a generic interface. The actual implementation uses ForwardDiff.jl
when available. Users can also provide their own Jacobian function.

# Arguments
- `f`: Function mapping R^n → R^m
- `x`: Point at which to compute Jacobian (vector of length n)

# Returns
- Jacobian matrix of shape (m, n)
"""
function jacobian end

# Default implementation using finite differences (fallback)
# This is slow and inaccurate - prefer AD-based implementations
function jacobian_fd(f, x::AbstractVector{T}; ε::T=sqrt(eps(T))) where {T}
    n = length(x)
    fx = f(x)
    m = length(fx)
    J = zeros(T, m, n)
    x_pert = copy(x)
    for j in 1:n
        x_pert[j] = x[j] + ε
        fx_pert = f(x_pert)
        J[:, j] = (fx_pert - fx) / ε
        x_pert[j] = x[j]
    end
    return J
end

####
#### Batch Jacobian Computation
####

"""
    batch_jacobians(f, xs::AbstractMatrix)

Compute Jacobians at multiple points in parallel.

# Arguments
- `f`: Function mapping R^D → R^D
- `xs`: Matrix of shape (T, D) where each row is a point

# Returns
- Array of shape (T, D, D) where result[t, :, :] is the Jacobian at xs[t, :]
"""
function batch_jacobians(f, xs::AbstractMatrix{T}, jacobian_fn=jacobian_fd) where {T}
    T_len, D = size(xs)
    Js = zeros(T, T_len, D, D)
    # This loop is parallelizable
    for t in 1:T_len
        Js[t, :, :] = jacobian_fn(f, xs[t, :])
    end
    return Js
end

####
#### Diagonal Extraction
####

"""
    jacobian_diagonal(f, x)

Compute only the diagonal of the Jacobian matrix.

More efficient than computing the full Jacobian when only diagonal is needed.
Uses forward-mode AD with standard basis vectors.

# Arguments
- `f`: Function mapping R^D → R^D (must have same input and output dimension)
- `x`: Point at which to compute Jacobian diagonal

# Returns
- Vector of length D containing diag(J)
"""
function jacobian_diagonal end

# Default implementation: compute full Jacobian and extract diagonal
function jacobian_diagonal_full(f, x::AbstractVector{T}, jacobian_fn=jacobian_fd) where {T}
    J = jacobian_fn(f, x)
    return diag(J)
end

"""
    batch_jacobian_diagonals(f, xs::AbstractMatrix)

Compute Jacobian diagonals at multiple points.

# Arguments
- `f`: Function mapping R^D → R^D
- `xs`: Matrix of shape (T, D) where each row is a point

# Returns
- Matrix of shape (T, D) where result[t, :] is the Jacobian diagonal at xs[t, :]
"""
function batch_jacobian_diagonals(f, xs::AbstractMatrix{T}, diag_fn=jacobian_diagonal_full) where {T}
    T_len, D = size(xs)
    diags = zeros(T, T_len, D)
    # This loop is parallelizable
    for t in 1:T_len
        diags[t, :] = diag_fn(f, xs[t, :])
    end
    return diags
end

####
#### JVP (Jacobian-Vector Product) Interface
####

"""
    jvp(f, x, v)

Compute the Jacobian-vector product J(x) * v where J is the Jacobian of f at x.

This is the fundamental operation for forward-mode AD. It computes the
directional derivative of f at x in direction v.

# Arguments
- `f`: Function mapping R^n → R^m
- `x`: Point at which to compute Jacobian
- `v`: Vector to multiply with Jacobian

# Returns
- Vector J(x) * v of length m
"""
function jvp end

# Default implementation using finite differences
function jvp_fd(f, x::AbstractVector{T}, v::AbstractVector{T}; ε::T=sqrt(eps(T))) where {T}
    return (f(x + ε * v) - f(x)) / ε
end

####
#### VJP (Vector-Jacobian Product) Interface
####

"""
    vjp(f, x, u)

Compute the vector-Jacobian product u' * J(x) where J is the Jacobian of f at x.

This is the fundamental operation for reverse-mode AD.

# Arguments
- `f`: Function mapping R^n → R^m
- `x`: Point at which to compute Jacobian
- `u`: Vector to multiply with Jacobian (from left)

# Returns
- Vector u' * J(x) of length n
"""
function vjp end

# Default implementation using finite differences (very slow, just for testing)
function vjp_fd(f, x::AbstractVector{T}, u::AbstractVector{T}; ε::T=sqrt(eps(T))) where {T}
    # vjp = J' * u, where J[i,j] = ∂f_i/∂x_j
    # (J' * u)[j] = sum_i J[i,j] * u[i] = sum_i u[i] * ∂f_i/∂x_j
    n = length(x)
    result = zeros(T, n)
    fx = f(x)
    x_pert = copy(x)
    for j in 1:n
        x_pert[j] = x[j] + ε
        fx_pert = f(x_pert)
        result[j] = dot(u, (fx_pert - fx) / ε)
        x_pert[j] = x[j]
    end
    return result
end

####
#### Hutchinson's Stochastic Diagonal Estimator
####

"""
    rademacher_vector(rng::AbstractRNG, n::Int, T::Type=Float64)

Generate a Rademacher random vector (entries ±1 with equal probability).

# Arguments
- `rng`: Random number generator
- `n`: Length of vector
- `T`: Element type

# Returns
- Vector of length n with entries ±1
"""
function rademacher_vector(rng::AbstractRNG, n::Int, ::Type{T}=Float64) where {T}
    return T(2) .* (rand(rng, n) .< 0.5) .- T(1)
end

rademacher_vector(n::Int, T::Type=Float64) = rademacher_vector(default_rng(), n, T)

"""
    hutchinson_diagonal(f, x, jvp_fn; rng=default_rng(), n_samples=1)

Estimate the diagonal of the Jacobian using Hutchinson's stochastic estimator.

The estimator uses the identity: diag(J) = E[z ⊙ (J * z)] where z is a
Rademacher random vector and ⊙ is elementwise multiplication.

This only requires JVP computations, not full Jacobian computation.

# Arguments
- `f`: Function mapping R^D → R^D
- `x`: Point at which to estimate Jacobian diagonal
- `jvp_fn`: Function computing JVP: jvp_fn(f, x, v) returns J(x) * v
- `rng`: Random number generator (default: default_rng())
- `n_samples`: Number of Monte Carlo samples (default: 1)

# Returns
- Vector of length D containing estimated diag(J)

# Reference
Hutchinson, M.F. (1990). "A stochastic estimator of the trace of the influence
matrix for Laplacian smoothing splines"
"""
function hutchinson_diagonal(
    f,
    x::AbstractVector{T},
    jvp_fn;
    rng::AbstractRNG=default_rng(),
    n_samples::Int=1,
) where {T}
    D = length(x)
    diag_estimate = zeros(T, D)

    for _ in 1:n_samples
        z = rademacher_vector(rng, D, T)
        Jz = jvp_fn(f, x, z)
        diag_estimate .+= z .* Jz
    end

    return diag_estimate ./ n_samples
end

"""
    batch_hutchinson_diagonals(f, xs::AbstractMatrix, jvp_fn; rng=default_rng(), n_samples=1)

Estimate Jacobian diagonals at multiple points using Hutchinson's estimator.

# Arguments
- `f`: Function mapping R^D → R^D
- `xs`: Matrix of shape (T, D) where each row is a point
- `jvp_fn`: Function computing JVP
- `rng`: Random number generator
- `n_samples`: Number of Monte Carlo samples per point

# Returns
- Matrix of shape (T, D) where result[t, :] is the estimated diagonal at xs[t, :]
"""
function batch_hutchinson_diagonals(
    f,
    xs::AbstractMatrix{T},
    jvp_fn;
    rng::AbstractRNG=default_rng(),
    n_samples::Int=1,
) where {T}
    T_len, D = size(xs)
    diags = zeros(T, T_len, D)

    # This loop is parallelizable
    for t in 1:T_len
        diags[t, :] = hutchinson_diagonal(f, xs[t, :], jvp_fn; rng=rng, n_samples=n_samples)
    end

    return diags
end

####
#### Hessian Diagonal (for leapfrog)
####

"""
    hessian_diagonal(grad_log_p, x)

Compute the diagonal of the Hessian of log p(x).

This is needed for Block Quasi-DEER applied to leapfrog integration.
The Hessian diagonal is the Jacobian diagonal of the gradient.

# Arguments
- `grad_log_p`: Gradient function ∇log p: R^D → R^D
- `x`: Point at which to compute Hessian diagonal

# Returns
- Vector of length D containing diag(∇²log p(x))
"""
function hessian_diagonal(grad_log_p, x::AbstractVector{T}, diag_fn=jacobian_diagonal_full) where {T}
    return diag_fn(grad_log_p, x)
end

"""
    batch_hessian_diagonals(grad_log_p, xs::AbstractMatrix)

Compute Hessian diagonals at multiple points.

# Arguments
- `grad_log_p`: Gradient function ∇log p: R^D → R^D
- `xs`: Matrix of shape (T, D) where each row is a point

# Returns
- Matrix of shape (T, D) where result[t, :] is the Hessian diagonal at xs[t, :]
"""
function batch_hessian_diagonals(grad_log_p, xs::AbstractMatrix{T}, diag_fn=jacobian_diagonal_full) where {T}
    return batch_jacobian_diagonals(grad_log_p, xs, diag_fn)
end

####
#### Utility: Linear Algebra
####

using LinearAlgebra: diag, dot
