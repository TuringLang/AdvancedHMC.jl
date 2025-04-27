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

# Convenient constructor
function DenseRiemannianMetric(size, ℓπ, initial_θ, λ, map=IdentityMap())
    _Hfunc = VecTargets.gen_hess(x -> -ℓπ(x), initial_θ) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H + λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(x -> -ℓπ(x), initial_θ; f=fstabilize)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x))

    _temp = Vector{Float64}(undef, first(size))

    return DenseRiemannianMetric(size, Gfunc, ∂G∂θfunc, map, _temp)
end

function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

#= possible integrate DI for AD-independent fisher information metric
function gen_∂G∂θ_rev(Vfunc, x; f=identity)
    _Hfunc = VecTargets.gen_hess(Vfunc, ReverseDiff.track.(x))
    Hfunc = x -> _Hfunc(x)[3]
    # QUES What's the best output format of this function?
    return x -> ReverseDiff.jacobian(x -> f(Hfunc(x)), x) # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end
=#

# Fisher information metric
function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    _Hfunc = gen_hess_fwd(Vfunc, x)
    Hfunc = x -> _Hfunc(x)[3]
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
    return out # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[((i - 1) * d + 1):(i * d), :] for i in 1:d)...; dims=3)
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
