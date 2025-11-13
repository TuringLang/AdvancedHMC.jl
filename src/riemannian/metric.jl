using AdvancedHMC: AbstractMetric
using LinearAlgebra: eigen, cholesky, Symmetric
import Base: eltype

# _randn is defined in utilities.jl which is included before this file

abstract type AbstractRiemannianMetric <: AbstractMetric end

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
function softabs(X, α=20.0)
    F = eigen(X) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * diagm(softabsλ) * Q', Q, λ, softabsλ
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
function DenseRiemannianMetric(size, G, ∂G∂θ, map=IdentityMap()) where {T<:AbstractFloat}
    _temp = Vector{Float64}(undef, size[1])
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp)
end

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.show(io::IO, dem::DenseRiemannianMetric) = print(io, "DenseRiemannianMetric(...)")

#function eltype(m::DenseRiemannianMetric)
#    return eltype(m._temp)
#end

eltype(::DenseRiemannianMetric{T}) where {T} = T

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric{T},
    kinetic,
    θ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    chol = cholesky(Symmetric(metric.map(metric.G(θ))))
    r = chol.L * r
    return r
end
