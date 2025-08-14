struct Hamiltonian{M<:AbstractMetric,K<:AbstractKinetic,Tlogπ,T∂logπ∂θ}
    metric::M
    kinetic::K
    ℓπ::Tlogπ
    ∂ℓπ∂θ::T∂logπ∂θ
end
function Base.show(io::IO, mime::MIME"text/plain", h::Hamiltonian)
    return print(
        io,
        "Hamiltonian with ",
        nameof(typeof(h.metric)),
        " and ",
        nameof(typeof(h.kinetic)),
    )
end

# By default we use Gaussian kinetic energy; also to ensure backward compatibility at the time this was introduced
function Hamiltonian(metric::AbstractMetric, ℓπ::Function, ∂ℓπ∂θ::Function)
    return Hamiltonian(metric, GaussianKinetic(), ℓπ, ∂ℓπ∂θ)
end

struct DualValue{
    V<:AbstractScalarOrVec{<:AbstractFloat},G<:AbstractVecOrMat{<:AbstractFloat}
}
    value::V    # cached value, e.g. logπ(θ)
    gradient::G # cached gradient, e.g. ∇logπ(θ)
    function DualValue(value::V, gradient::G) where {V,G}
        # Check consistence
        if value isa AbstractFloat
            # If `value` is a scalar, `gradient` is a vector
            @assert gradient isa AbstractVector "`typeof(gradient)`: $(typeof(gradient))"
        else
            # If `value` is a vector, `gradient` is a matrix
            @assert gradient isa AbstractMatrix "`typeof(gradient)`: $(typeof(gradient))"
        end
        return new{V,G}(value, gradient)
    end
end

function Base.similar(dv::DualValue{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat}
    return DualValue(zeros(T, size(dv.value)...), zeros(T, size(dv.gradient)...))
end

# `∂H∂θ` now returns `(logprob, -∂ℓπ∂θ)`
function ∂H∂θ(h::Hamiltonian, θ::AbstractVecOrMat)
    res = h.∂ℓπ∂θ(θ)
    return DualValue(res[1], -res[2])
end

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric,<:GaussianKinetic}, r::AbstractVecOrMat) = copy(r)
function ∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric,<:GaussianKinetic}, r::AbstractVecOrMat)
    (; M⁻¹) = h.metric
    axes_M⁻¹ = __axes(M⁻¹)
    axes_r = __axes(r)
    (first(axes_M⁻¹) !== first(axes_r)) && throw(
        ArgumentError("AxesMismatch: M⁻¹ has axes $(axes_M⁻¹) but r has axes $(axes_r)")
    )
    return M⁻¹ .* r
end
function ∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric,<:GaussianKinetic}, r::AbstractVecOrMat)
    (; M⁻¹) = h.metric
    axes_M⁻¹ = __axes(M⁻¹)
    axes_r = __axes(r)
    (last(axes_M⁻¹) !== first(axes_r)) && throw(
        ArgumentError("AxesMismatch: M⁻¹ has axes $(axes_M⁻¹) but r has axes $(axes_r)")
    )
    return M⁻¹ * r
end

function ∂H∂r(
    h::Hamiltonian{<:RankUpdateEuclideanMetric,<:GaussianKinetic}, r::AbstractVecOrMat
)
    (; M⁻¹) = h.metric
    axes_M⁻¹ = __axes(M⁻¹)
    axes_r = __axes(r)
    (first(axes_M⁻¹) !== first(axes_r)) && throw(
        ArgumentError("AxesMismatch: M⁻¹ has axes $(axes_M⁻¹) but r has axes $(axes_r)")
    )
    return M⁻¹ * r
end

# TODO (kai) make the order of θ and r consistent with neg_energy
# TODO (kai) add stricter types to block hamiltonian.jl#L37 from working on unknown metric/kinetic
# The gradient of a position-dependent Hamiltonian system depends on both θ and r. 
∂H∂θ(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂θ(h, θ)
∂H∂r(h::Hamiltonian, θ::AbstractVecOrMat, r::AbstractVecOrMat) = ∂H∂r(h, r)

struct PhasePoint{T<:AbstractVecOrMat{<:AbstractFloat},V<:DualValue}
    θ::T  # Position variables / model parameters.
    r::T  # Momentum variables
    ℓπ::V # Cached neg potential energy for the current θ.
    ℓκ::V # Cached neg kinetic energy for the current r.
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T,V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓκ.gradient)
        if !isfinite(ℓπ)
            ℓπ = DualValue(
                map(v -> isfinite(v) ? v : oftype(v, -Inf), ℓπ.value), ℓπ.gradient
            )
        end
        if !isfinite(ℓκ)
            ℓκ = DualValue(
                map(v -> isfinite(v) ? v : oftype(v, -Inf), ℓκ.value), ℓκ.gradient
            )
        end
        return new{T,V}(θ, r, ℓπ, ℓκ)
    end
end

function Base.similar(z::PhasePoint{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat}
    return PhasePoint(
        zeros(T, size(z.θ)...), zeros(T, size(z.r)...), similar(z.ℓπ), similar(z.ℓκ)
    )
end

function phasepoint(
    h::Hamiltonian, θ::T, r::T; ℓπ=∂H∂θ(h, θ), ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r))
) where {T<:AbstractVecOrMat}
    return PhasePoint(θ, r, ℓπ, ℓκ)
end

# If position variable and momentum variable are in different containers,
# move the momentum variable to that of the position variable.
# This is needed for AHMC to work with CuArrays and other Arrays (without depending on it).
function phasepoint(
    h::Hamiltonian,
    θ::T1,
    _r::T2;
    r=safe_rsimilar(θ, _r),
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r)),
) where {T1<:AbstractVecOrMat,T2<:AbstractVecOrMat}
    return PhasePoint(θ, r, ℓπ, ℓκ)
end
# ensures compatibility with ComponentArrays
function safe_rsimilar(θ, _r)
    r = similar(θ)
    copyto!(r, _r)
    return r
end

Base.isfinite(v::DualValue) = all(isfinite, v.value) && all(isfinite, v.gradient)
Base.isfinite(z::PhasePoint) = isfinite(z.ℓπ) && isfinite(z.ℓκ)

###
### Negative energy (or log probability) functions.
### NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r.
###

neg_energy(z::PhasePoint) = z.ℓπ.value + z.ℓκ.value

neg_energy(h::Hamiltonian, θ::AbstractVecOrMat) = h.ℓπ(θ)

# GaussianKinetic

function neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractVector}
    return -sum(abs2, r) / 2
end

function neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractMatrix}
    return -vec(sum(abs2, r; dims=1)) / 2
end

function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractVector}
    return -sum(abs2.(r) .* h.metric.M⁻¹) / 2
end

function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractMatrix}
    return -vec(sum(abs2.(r) .* h.metric.M⁻¹; dims=1)) / 2
end

function neg_energy(
    h::Hamiltonian{<:DenseEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return -dot(r, h.metric._temp) / 2
end

function neg_energy(
    h::Hamiltonian{<:RankUpdateEuclideanMetric,<:GaussianKinetic}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    M⁻¹ = h.metric.M⁻¹
    return -r' * M⁻¹ * r / 2
end

energy(args...) = -neg_energy(args...)

####
#### Momentum refreshment
####

function phasepoint(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian,
) where {T<:Real}
    return phasepoint(h, θ, rand_momentum(rng, h.metric, h.kinetic, θ))
end

abstract type AbstractMomentumRefreshment end

"Completly resample new momentum."
struct FullMomentumRefreshment <: AbstractMomentumRefreshment end

function refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
)
    return phasepoint(h, z.θ, rand_momentum(rng, h.metric, h.kinetic, z.θ))
end

"""
$(TYPEDEF)
Partial momentum refreshment with refresh rate `α`.

# Fields
$(TYPEDFIELDS)

See equation (5.19) [1]

    r' = α⋅r + sqrt(1-α²)⋅G

where r is the momentum and G is a Gaussian random variable.

## References

1. Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov chain monte carlo 2.11 (2011): 2.
"""
struct PartialMomentumRefreshment{F<:AbstractFloat} <: AbstractMomentumRefreshment
    α::F
end

function refresh(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    ref::PartialMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
)
    return phasepoint(
        h,
        z.θ,
        ref.α * z.r + sqrt(1 - ref.α^2) * rand_momentum(rng, h.metric, h.kinetic, z.θ),
    )
end
