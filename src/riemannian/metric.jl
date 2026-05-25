using LinearAlgebra: LinearAlgebra

####
#### Riemannian Metric Types
####

"""
Abstract type for Riemannian (position-dependent) metrics.

Subtypes must implement:
- `metric_eval(metric, θ)` - evaluate metric at position θ
- `metric_sensitivity(metric, θ)` - compute ∂P/∂θ where P is the "pre-metric"
  (G itself for RiemannianMetric, or H the Hessian for SoftAbsRiemannianMetric)
"""
abstract type AbstractRiemannianMetric <: AbstractMetric end

####
#### SoftAbsEval - cached eigendecomposition for SoftAbs metrics
####

"""
    SoftAbsEval{T}

Cached result of evaluating a SoftAbs metric at a position θ.
Stores eigendecomposition and precomputed matrices for efficient gradient computation.

# Fields
- `Q`: Eigenvectors (orthogonal matrix)
- `softabsλ`: Transformed eigenvalues: λᵢ * coth(α * λᵢ)
- `J`: Jacobian matrix encoding the derivative of softabs (divided difference formula)
- `M_logdet`: Precomputed matrix Q * (R .* J) * Q' for logdet gradient
"""
struct SoftAbsEval{T<:AbstractFloat}
    Q::Matrix{T}
    softabsλ::Vector{T}
    J::Matrix{T}
    M_logdet::Matrix{T}
end

# Standard operations for SoftAbsEval
function Base.:\(G::SoftAbsEval, p::AbstractVector)
    return G.Q * ((G.Q' * p) ./ G.softabsλ)
end

function LinearAlgebra.logdet(G::SoftAbsEval)
    return sum(log, G.softabsλ)
end

"""
    unwhiten(G::SoftAbsEval, z)

Transform z ~ N(0, I) to sample from N(0, G).
"""
function unwhiten(G::SoftAbsEval, z::AbstractVector)
    return G.Q * (sqrt.(G.softabsλ) .* z)
end

####
#### RiemannianMetric - for user-provided PD metrics
####

"""
    RiemannianMetric{T, TG, T∂G}

Riemannian metric where the user provides a function returning a positive-definite
matrix (or AbstractPDMat subtype). `T` is the element type used for momentum sampling
and defaults to `$(DEFAULT_FLOAT_TYPE)` if not specified.

For best performance, return an `AbstractPDMat` so that the kinetic and logdet gradients can
reuse the stored Cholesky factor instead of refactorising on every call.

If you need to stabilise `G` (e.g. by adding `λI`), do so *inside* the `PDMat` constructor 
since `PDMat(G) + λ*I` and other variants may fall back to returning Symmetric or even dense
matrices.

# Fields
- `size`: Tuple{Int} giving the dimension
- `calc_G`: Function θ → G(θ), returns a positive-definite matrix
- `calc_∂G∂θ`: Function θ → ∂G/∂θ, returns Array{T,3} of shape (d, d, d)

# Example
```julia
# Simple Fisher information metric
calc_G = θ -> PDMat(fisher_information(θ))
calc_∂G∂θ = θ -> ForwardDiff.jacobian(θ -> vec(fisher_information(θ)), θ) |> reshape_∂G∂θ
metric = RiemannianMetric((d,), calc_G, calc_∂G∂θ)  # T = Float64
metric_f32 = RiemannianMetric{Float32}((d,), calc_G, calc_∂G∂θ)
```
"""
struct RiemannianMetric{T<:AbstractFloat,TG,T∂G} <: AbstractRiemannianMetric
    size::Tuple{Int}
    calc_G::TG       # θ → Matrix or AbstractPDMat
    calc_∂G∂θ::T∂G   # θ → Array{T,3}
end

function RiemannianMetric{T}(
    size::Tuple{Int}, calc_G::TG, calc_∂G∂θ::T∂G
) where {T<:AbstractFloat,TG,T∂G}
    return RiemannianMetric{T,TG,T∂G}(size, calc_G, calc_∂G∂θ)
end

function RiemannianMetric(size::Tuple{Int}, calc_G, calc_∂G∂θ)
    return RiemannianMetric{DEFAULT_FLOAT_TYPE}(size, calc_G, calc_∂G∂θ)
end

Base.size(m::RiemannianMetric) = m.size
Base.size(m::RiemannianMetric, dim::Int) = m.size[dim]
Base.eltype(::RiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::RiemannianMetric)
    return print(io, "RiemannianMetric(size=", m.size, ")")
end

# Interface implementations for RiemannianMetric
metric_eval(m::RiemannianMetric, θ) = m.calc_G(θ)
metric_sensitivity(m::RiemannianMetric, θ) = m.calc_∂G∂θ(θ)

####
#### SoftAbsRiemannianMetric - for Hessian-based metrics with SoftAbs regularization
####

"""
    SoftAbsRiemannianMetric{T, TH, T∂H}

Riemannian metric based on the SoftAbs transformation of a Hessian.
The Hessian may not be positive-definite; the SoftAbs transformation
G = Q * diag(λ * coth(α*λ)) * Q' guarantees positive-definiteness.

# Fields
- `size`: Tuple{Int} giving the dimension
- `calc_H`: Function θ → H(θ), returns the Hessian matrix (the "pre-metric")
- `calc_∂H∂θ`: Function θ → ∂H/∂θ, returns Array{T,3} of shape (d, d, d)
- `α`: SoftAbs regularization parameter (larger = closer to |λ|)
- `canonicalize`: If `true`, eigenvector signs are fixed after each `eigen` call so that
  the largest-magnitude element of each column is positive. This makes momentum sampling
  via `unwhiten` reproducible across BLAS/LAPACK implementations that may return the same
  eigenvectors with opposite sign conventions (e.g. OpenBLAS on x86 vs arm64). Has no
  effect on the metric matrix G or any gradient — those are sign-invariant. Default `false`.

# References
- Betancourt, M. "A general metric for Riemannian manifold Hamiltonian Monte Carlo" (2012)
"""
struct SoftAbsRiemannianMetric{T<:AbstractFloat,TH,T∂H} <: AbstractRiemannianMetric
    size::Tuple{Int}
    calc_H::TH       # θ → Hessian matrix (pre-metric)
    calc_∂H∂θ::T∂H   # θ → Array{T,3}
    α::T
    canonicalize::Bool
end

function SoftAbsRiemannianMetric(
    size::Tuple{Int}, calc_H::TH, calc_∂H∂θ::T∂H, α::T; canonicalize::Bool=false
) where {T<:AbstractFloat,TH,T∂H}
    return SoftAbsRiemannianMetric{T,TH,T∂H}(size, calc_H, calc_∂H∂θ, α, canonicalize)
end

Base.size(m::SoftAbsRiemannianMetric) = m.size
Base.size(m::SoftAbsRiemannianMetric, dim::Int) = m.size[dim]
Base.eltype(::SoftAbsRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::SoftAbsRiemannianMetric)
    print(io, "SoftAbsRiemannianMetric(size=", m.size, ", α=", m.α)
    m.canonicalize && print(io, ", canonicalize=true")
    return print(io, ")")
end

# Compile-time cutoffs for the SoftAbs stability switches. `@generated` ensures
# `eps(T)^(1//n)` is evaluated once per specialisation, not on every call.
@generated _xcothx_cutoff(::Type{T}) where {T<:AbstractFloat} = eps(T)^(1//6)
@generated _make_J_cutoff(::Type{T}) where {T<:AbstractFloat} = eps(T)^(1//3)

# Branch implementations of x·coth(x). _exact is correct away from zero; _taylor
# is correct near zero. _xcothx dispatches between them at _xcothx_cutoff(T).
@inline _xcothx_taylor(x::T) where {T<:AbstractFloat} = one(T) + x^2 / 3 - x^4 / 45
@inline _xcothx_exact(x::T) where {T<:AbstractFloat} = x * coth(x)

"""
    _xcothx(x)

Compute `x * coth(x)` (i.e. `α * softabs(λ)` for `x = α*λ`) in a way that is
numerically stable as `x → 0`.

Naively, `x * coth(x)` evaluates to `0 * Inf = NaN` at exactly `x = 0`, even though
the true limit is `1`. We switch to the two-term Taylor expansion
`1 + x²/3 − x⁴/45` below `|x| < eps(T)^(1//6)`, whose truncation error
(`O(x⁶) ≈ eps`) is at machine precision by the time we hit the switch.
"""
function _xcothx(x::T) where {T<:AbstractFloat}
    return abs(x) < _xcothx_cutoff(T) ? _xcothx_taylor(x) : _xcothx_exact(x)
end

# Branch implementations of d/dx[x·coth(x)] = coth(x) − x·csch(x)².
@inline _xcothx_deriv_taylor(x::T) where {T<:AbstractFloat} = T(2) / 3 * x - T(4) / 45 * x^3
@inline _xcothx_deriv_exact(x::T) where {T<:AbstractFloat} = coth(x) - x * csch(x)^2

"""
    _xcothx_deriv(x)

Compute the derivative `coth(x) − x * csch(x)²` of `x * coth(x)`, stably as `x → 0`.

Naively, both `coth(x)` and `x * csch(x)²` behave like `1/x` near zero, so direct
subtraction suffers catastrophic cancellation (relative error `~eps/x²`); at `x = 0`
the result is `Inf − Inf = NaN`. The Taylor expansion is `2x/3 − 4x³/45 + O(x⁵)`.
Balancing Taylor truncation (`~x⁴` relative) against cancellation (`~eps/x²` relative)
gives the optimal switch at `|x| < eps(T)^(1//6)`, yielding ~13 digits across the range.
"""
function _xcothx_deriv(x::T) where {T<:AbstractFloat}
    return abs(x) < _xcothx_cutoff(T) ? _xcothx_deriv_taylor(x) : _xcothx_deriv_exact(x)
end

"""
    make_J(λ, α)

Construct the J matrix for softabs gradient computation.
J encodes the derivative of the softabs transformation using the divided difference formula.

For `λᵢ` well separated from `λⱼ`:
    J[i,j] = (softabs(λᵢ) − softabs(λⱼ)) / (λᵢ − λⱼ)
For `λᵢ ≈ λⱼ` (including the diagonal):
    J[i,j] = d/dλ [λ coth(αλ)] |_{λ = (λᵢ + λⱼ)/2}

The branches are switched when `|α(λᵢ − λⱼ)| < eps(T)^(1//3)`: at that point the
divided-difference cancellation (`~eps/(αδ)`) and the midpoint-rule truncation
(`~(αδ)²`) balance, each contributing `~eps^(2/3)` relative error.

# References
- Betancourt (2012)
"""
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    deg_tol = _make_J_cutoff(T)
    @inbounds for j in 1:d, i in 1:d
        xi = α * λ[i]
        xj = α * λ[j]
        if abs(xi - xj) < deg_tol
            # Diagonal or near-degenerate: derivative at the midpoint (stable at x = 0).
            J[i, j] = _xcothx_deriv((xi + xj) / 2)
        else
            # Divided difference written in terms of α·λ so _xcothx handles λ = 0 safely.
            J[i, j] = (_xcothx(xi) - _xcothx(xj)) / (xi - xj)
        end
    end
    return J
end

"""
    metric_eval(m::SoftAbsRiemannianMetric, θ)

Evaluate SoftAbs metric at position θ, returning a `SoftAbsEval` with cached matrices.
"""
function metric_eval(m::SoftAbsRiemannianMetric{T}, θ) where {T}
    H = m.calc_H(θ)
    F = eigen(Symmetric(H))
    λ = F.values
    Q = F.vectors

    if m.canonicalize
        @inbounds for j in axes(Q, 2)
            col = view(Q, :, j)
            if col[argmax(abs.(col))] < 0
                col .*= -1
            end
        end
    end

    # SoftAbs transformation: G = Q * diag(softabsλ) * Q'.
    # Use _xcothx to avoid `0 * Inf = NaN` at exactly λ = 0 (limit is 1/α).
    softabsλ = _xcothx.(m.α .* λ) ./ m.α

    # Compute J matrix for gradient chain rule
    J = make_J(λ, m.α)

    # Precompute M_logdet = Q * (R .* J) * Q' where R = diag(1 ./ softabsλ)
    # This is used for: ∂log|G|/∂θᵢ = 0.5 * tr(M_logdet * ∂H/∂θᵢ)
    R = Diagonal(one(T) ./ softabsλ)
    M_logdet = Q * (R .* J) * Q'

    return SoftAbsEval(Q, softabsλ, J, M_logdet)
end

metric_sensitivity(m::SoftAbsRiemannianMetric, θ) = m.calc_∂H∂θ(θ)

####
#### Gradient matrices for unified computation
####

"""
    logdet_grad_matrix(G)

Return the matrix M such that ∂log|G|/∂θᵢ = 0.5 * tr(M * ∂P/∂θᵢ), where P is the
"pre-metric" (G itself for RiemannianMetric, or H the Hessian for SoftAbsRiemannianMetric).

For dense matrices: M = G⁻¹
For SoftAbsEval: M = Q * (R .* J) * Q' (precomputed in metric_eval)

The J matrix in SoftAbsEval absorbs the chain rule through the softabs transformation,
so the same formula works with ∂H/∂θ instead of ∂G/∂θ.
"""
logdet_grad_matrix(G::SoftAbsEval) = G.M_logdet
logdet_grad_matrix(G::AbstractMatrix) = inv(G)

"""
    kinetic_grad_matrix(G, r)

Return the matrix M such that ∂(r'G⁻¹r)/∂θᵢ = -tr(M * ∂P/∂θᵢ), where P is the
"pre-metric" (G itself for RiemannianMetric, or H the Hessian for SoftAbsRiemannianMetric).

For dense matrices: M = (G⁻¹r)(G⁻¹r)' (rank-1 outer product)
For SoftAbsEval: M = Q * D * J * D * Q' where D = diag((Q'r) ./ softabsλ)

For SoftAbsEval, the J matrix absorbs the chain rule through softabs, allowing
the gradient to be computed with respect to ∂H/∂θ rather than ∂G/∂θ. This avoids
recomputing J for each value of r during fixed-point iterations.
"""
function kinetic_grad_matrix(G::SoftAbsEval, r::AbstractVector)
    # D = diag((Q'r) ./ softabsλ)
    d = (G.Q' * r) ./ G.softabsλ
    D = Diagonal(d)
    return G.Q * D * G.J * D * G.Q'
end

function kinetic_grad_matrix(G::AbstractMatrix, r::AbstractVector)
    v = G \ r
    return v * v'  # Rank-1 outer product
end

####
#### Momentum sampling
####

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::AbstractRiemannianMetric,
    ::GaussianKinetic,
    θ::AbstractVecOrMat,
)
    G = metric_eval(metric, θ)
    z = _randn(rng, eltype(metric), size(metric)...)
    return unwhiten(G, z)
end

# unwhiten for regular matrices (PDMat or dense)
function unwhiten(G::AbstractMatrix, z::AbstractVector)
    # G = L * L', so sample = L * z where L = chol(G).L
    chol = cholesky(Symmetric(G))
    return chol.L * z
end

####
#### Deprecated API forwards: DenseRiemannianMetric → {RiemannianMetric, SoftAbsRiemannianMetric}
####
#### `IdentityMap` and `SoftAbsMap` are retained as minimal tag types so the old
#### `DenseRiemannianMetric(size, G, ∂G∂θ, map)` signature still parses; they have no
#### runtime role beyond dispatching the @deprecate forwards below.

struct IdentityMap end

struct SoftAbsMap{T}
    α::T
end

@deprecate(
    DenseRiemannianMetric(size::Tuple{Int}, G, ∂G∂θ), RiemannianMetric(size, G, ∂G∂θ),
)
@deprecate(
    DenseRiemannianMetric(size::Tuple{Int}, G, ∂G∂θ, ::IdentityMap),
    RiemannianMetric(size, G, ∂G∂θ),
)
@deprecate(
    DenseRiemannianMetric(size::Tuple{Int}, G, ∂G∂θ, map::SoftAbsMap),
    SoftAbsRiemannianMetric(size, G, ∂G∂θ, map.α),
)
