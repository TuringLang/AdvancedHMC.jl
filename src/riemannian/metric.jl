abstract type AbstractRiemannianMetric <: AbstractMetric end

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
function softabs(X, α = 20.0)
    F = eigen(
        Symmetric(X)    # NOTE ~Symmetric~ is needed to avid eigen returns complex numbers
    ) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * Diagonal(softabsλ) * Q', Q, λ, softabsλ
end

function softabs(X::T, α = 20.0) where {T <: Diagonal}
    Q = I
    λ = X.diag
    softabsλ = λ .* coth.(α * λ)
    return Q * Diagonal(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

struct DenseRiemannianMetric{
    T,
    TM<:AbstractHessianMap,
    A<:Union{Tuple{Int},Tuple{Int,Int}},
    AV<:AbstractVecOrMat{T},
    TG,
    T∂G∂θ,
} <: AbstractRiemannianMetric
    size::A
    G::TG # TODO store G⁻¹ here instead
    ∂G∂θ::T∂G∂θ
    map::TM
    _temp::AV
end

# TODO Make dense mass matrix support matrix-mode parallel
function DenseRiemannianMetric(size, G, ∂G∂θ, map = IdentityMap())
    _temp = Vector{Float64}(undef, size[1])
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.show(io::IO, dem::DenseRiemannianMetric) = print(io, "DenseRiemannianMetric(...)")

function _rand(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric{T},
    kinetic::Union{GaussianKinetic, <:AbstractRelativisticKinetic{T}},
    θ::AbstractVecOrMat,
) where {T}
    r = _rand(rng, UnitEuclideanMetric(size(metric)), kinetic)
    M⁻¹ = inv(metric.map(metric.G(θ)))
    cholM⁻¹ = cholesky(Symmetric(M⁻¹)).U
    ldiv!(cholM⁻¹, r)
    return r
end

Base.rand(rng::AbstractRNG, metric::AbstractRiemannianMetric, kinetic::AbstractKinetic, θ::AbstractVecOrMat) =
    _rand(rng, metric, kinetic, θ)
Base.rand(
    rng::AbstractVector{<:AbstractRNG},
    metric::AbstractRiemannianMetric,
    kinetic::AbstractKinetic,
    θ::AbstractVecOrMat,
) = _rand(rng, metric, kinetic, θ)