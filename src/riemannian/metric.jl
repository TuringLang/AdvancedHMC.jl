abstract type AbstractRiemannianMetric <: AbstractMetric end

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

function softabs(X, α=20.0)
    F = eigen(X) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * diagm(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
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
function DenseRiemannianMetric(size, G, ∂G∂θ, map=IdentityMap())
    _temp = Vector{Float64}(undef, first(size))
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
function Base.show(io::IO, drm::DenseRiemannianMetric)
    return print(io, "DenseRiemannianMetric$(drm.size) with $(drm.map) metric")
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric{T},
    kinetic,
    θ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    G⁻¹ = inv(metric.map(metric.G(θ)))
    chol = cholesky(Symmetric(G⁻¹))
    ldiv!(chol.U, r)
    return r
end
