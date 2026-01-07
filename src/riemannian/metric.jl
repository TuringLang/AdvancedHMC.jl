import LinearAlgebra: logdet

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

function logdet(G::SoftAbsEval)
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
    RiemannianMetric{TG, T∂G}

Riemannian metric where the user provides a function returning a positive-definite
matrix (or AbstractPDMat subtype).

# Fields
- `size`: Tuple{Int} giving the dimension
- `calc_G`: Function θ → G(θ), returns a positive-definite matrix
- `calc_∂G∂θ`: Function θ → ∂G/∂θ, returns Array{T,3} of shape (d, d, d)

# Example
```julia
# Simple Fisher information metric
calc_G = θ -> PDMat(fisher_information(θ))
calc_∂G∂θ = θ -> ForwardDiff.jacobian(θ -> vec(fisher_information(θ)), θ) |> reshape_∂G∂θ
metric = RiemannianMetric((d,), calc_G, calc_∂G∂θ)
```
"""
struct RiemannianMetric{TG,T∂G} <: AbstractRiemannianMetric
    size::Tuple{Int}
    calc_G::TG       # θ → Matrix or AbstractPDMat
    calc_∂G∂θ::T∂G   # θ → Array{T,3}
end

Base.size(m::RiemannianMetric) = m.size
Base.size(m::RiemannianMetric, dim::Int) = m.size[dim]

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

# References
- Betancourt, M. "A general metric for Riemannian manifold Hamiltonian Monte Carlo" (2012)
"""
struct SoftAbsRiemannianMetric{T<:AbstractFloat,TH,T∂H} <: AbstractRiemannianMetric
    size::Tuple{Int}
    calc_H::TH       # θ → Hessian matrix (pre-metric)
    calc_∂H∂θ::T∂H   # θ → Array{T,3}
    α::T
end

Base.size(m::SoftAbsRiemannianMetric) = m.size
Base.size(m::SoftAbsRiemannianMetric, dim::Int) = m.size[dim]
Base.eltype(::SoftAbsRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::SoftAbsRiemannianMetric)
    return print(io, "SoftAbsRiemannianMetric(size=", m.size, ", α=", m.α, ")")
end

"""
    make_J(λ, α)

Construct the J matrix for softabs gradient computation.
J encodes the derivative of the softabs transformation using the divided difference formula.

For i ≠ j: J[i,j] = (softabs(λᵢ) - softabs(λⱼ)) / (λᵢ - λⱼ)
For i = j: J[i,i] = d/dλ [λ coth(αλ)] = coth(αλ) - αλ csch²(αλ)

# References
- Betancourt (2012)
"""
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    @inbounds for i in 1:d, j in 1:d
        if λ[i] == λ[j]
            # Derivative case (diagonal or degenerate eigenvalues)
            # d/dλ [λ coth(αλ)] = coth(αλ) - αλ csch²(αλ)
            J[i, j] = coth(α * λ[i]) - α * λ[i] * csch(α * λ[i])^2
        else
            # Divided difference
            J[i, j] = (λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j])
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

    # SoftAbs transformation: G = Q * diag(softabsλ) * Q'
    softabsλ = λ .* coth.(m.α .* λ)

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
    T = eltype(metric) === Any ? eltype(θ) : eltype(metric)
    z = _randn(rng, T, size(metric)...)
    return unwhiten(G, z)
end

# unwhiten for regular matrices (PDMat or dense)
function unwhiten(G::AbstractMatrix, z::AbstractVector)
    # G = L * L', so sample = L * z where L = chol(G).L
    chol = cholesky(Symmetric(G))
    return chol.L * z
end

# eltype for RiemannianMetric (needed for rand_momentum)
Base.eltype(::RiemannianMetric) = Any  # Will use eltype(θ) as fallback

####
#### Deprecated types (for backward compatibility)
####

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end
(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

function softabs(X, α=20.0)
    F = eigen(Symmetric(X))
    Q = F.vectors
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * Diagonal(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

"""
    DenseRiemannianMetric (deprecated)

Use `RiemannianMetric` or `SoftAbsRiemannianMetric` instead.
"""
struct DenseRiemannianMetric{
    T,
    TM<:AbstractHessianMap,
    A<:Union{Tuple{Int},Tuple{Int,Int}},
    AV<:AbstractVecOrMat{T},
    TG,
    T∂G∂θ,
} <: AbstractRiemannianMetric
    size::A
    G::TG
    ∂G∂θ::T∂G∂θ
    map::TM
    _temp::AV
end

function DenseRiemannianMetric(size, G, ∂G∂θ, map=IdentityMap())
    Base.depwarn(
        "DenseRiemannianMetric is deprecated. Use RiemannianMetric (for IdentityMap) or SoftAbsRiemannianMetric (for SoftAbsMap) instead.",
        :DenseRiemannianMetric,
    )
    _temp = Vector{Float64}(undef, first(size))
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.eltype(::DenseRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, drm::DenseRiemannianMetric)
    return print(
        io,
        "DenseRiemannianMetric",
        drm.size,
        " with ",
        nameof(typeof(drm.map)),
        " (deprecated)",
    )
end

# metric_eval and metric_sensitivity for deprecated DenseRiemannianMetric
function metric_eval(m::DenseRiemannianMetric{T,<:IdentityMap}, θ) where {T}
    return m.G(θ)
end

function metric_eval(m::DenseRiemannianMetric{T,<:SoftAbsMap}, θ) where {T}
    H = m.G(θ)
    F = eigen(Symmetric(H))
    λ = F.values
    Q = F.vectors
    softabsλ = λ .* coth.(m.map.α .* λ)
    J = make_J(λ, m.map.α)
    R = Diagonal(one(T) ./ softabsλ)
    M_logdet = Q * (R .* J) * Q'
    return SoftAbsEval(Q, softabsλ, J, M_logdet)
end

metric_sensitivity(m::DenseRiemannianMetric, θ) = m.∂G∂θ(θ)
