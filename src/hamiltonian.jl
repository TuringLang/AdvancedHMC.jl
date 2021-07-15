# TODO: add a type for kinetic energy

struct Hamiltonian{M<:AbstractMetric, Tlogπ, T∂logπ∂θ}
    metric::M
    ℓπ::Tlogπ
    ∂ℓπ∂θ::T∂logπ∂θ
end
Base.show(io::IO, h::Hamiltonian) = print(io, "Hamiltonian(metric=$(h.metric))")

struct DualValue{V<:AbstractScalarOrVec{<:AbstractFloat}, G<:AbstractVecOrMat{<:AbstractFloat}}
    value::V    # cached value, e.g. logπ(θ)
    gradient::G # cached gradient, e.g. ∇logπ(θ)
    function DualValue(value::V, gradient::G) where {V, G}
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

Base.similar(dv::DualValue{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat} = 
    DualValue(zeros(T, size(dv.value)...), zeros(T, size(dv.gradient)...))

# `∂H∂θ` now returns `(logprob, -∂ℓπ∂θ)`
function ∂H∂θ(h::Hamiltonian, θ::AbstractVecOrMat)
    res = h.∂ℓπ∂θ(θ)
    return DualValue(res[1], -res[2])
end

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVecOrMat) = copy(r)
∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVecOrMat) = h.metric.M⁻¹ .* r
∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVecOrMat) = h.metric.M⁻¹ * r

struct PhasePoint{T<:AbstractVecOrMat{<:AbstractFloat}, V<:DualValue}
    θ::T  # Position variables / model parameters.
    r::T  # Momentum variables
    ℓπ::V # Cached neg potential energy for the current θ.
    ℓκ::V # Cached neg kinect energy for the current r.
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T, V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓπ.gradient)
        if any(isfinite.((θ, r, ℓπ, ℓκ)) .== false)
            @warn "The current proposal will be rejected due to numerical error(s)." isfinite.((θ, r, ℓπ, ℓκ))
            # NOTE eltype has to be inlined to avoid type stability issue; see #267
            ℓπ = DualValue(
                map(v -> isfinite(v) ? v : -eltype(T)(Inf), ℓπ.value),
                ℓπ.gradient
            )
            ℓκ = DualValue(
                map(v -> isfinite(v) ? v : -eltype(T)(Inf), ℓκ.value),
                ℓκ.gradient
            )
        end
        new{T,V}(θ, r, ℓπ, ℓκ)
    end
end

Base.similar(z::PhasePoint{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat} = 
    PhasePoint(
        zeros(T, size(z.θ)...), 
        zeros(T, size(z.r)...), 
        similar(z.ℓπ), 
        similar(z.ℓκ),
    )

phasepoint(
    h::Hamiltonian,
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r))
) where {T<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

# If position variable and momentum variable are in different containers,
# move the momentum variable to that of the position variable.
# This is needed for AHMC to work with CuArrays and other Arrays (without depending on it).
phasepoint(
    h::Hamiltonian,
    θ::T1,
    _r::T2;
    r=safe_rsimilar(θ, _r),
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r))
) where {T1<:AbstractVecOrMat,T2<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)
# ensures compatibility with ComponentArrays
function safe_rsimilar(θ, _r)
    r = similar(θ)
    copyto!(r, _r)
    r
end


Base.isfinite(v::DualValue) = all(isfinite, v.value) && all(isfinite, v.gradient)
Base.isfinite(v::AbstractVecOrMat) = all(isfinite, v)
Base.isfinite(z::PhasePoint) = isfinite(z.ℓπ) && isfinite(z.ℓκ)

###
### Negative energy (or log probability) functions.
### NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r.
###

neg_energy(z::PhasePoint) = z.ℓπ.value + z.ℓκ.value

neg_energy(h::Hamiltonian, θ::AbstractVecOrMat) = h.ℓπ(θ)

neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = -sum(abs2, r) / 2

neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractMatrix} = -vec(sum(abs2, r; dims=1)) / 2

neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = -sum(abs2.(r) .* h.metric.M⁻¹) / 2

neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractMatrix} = -vec(sum(abs2.(r) .* h.metric.M⁻¹; dims=1) ) / 2

function neg_energy(
    h::Hamiltonian{<:DenseEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVecOrMat}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return -dot(r, h.metric._temp) / 2
end

energy(args...) = -neg_energy(args...)

####
#### Momentum refreshment
####

phasepoint(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian
) where {T<:Real} = phasepoint(h, θ, rand(rng, h.metric))

abstract type AbstractMomentumRefreshment end

"Completly resample new momentum."
struct FullMomentumRefreshment <: AbstractMomentumRefreshment end

refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::FullMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(h, z.θ, rand(rng, h.metric))

"""
Partial momentum refreshment with refresh rate `α`.

## References

1. Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov chain monte carlo 2.11 (2011): 2.
"""
struct PartialMomentumRefreshment{F<:AbstractFloat} <: AbstractMomentumRefreshment
    α::F
end

refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ref::PartialMomentumRefreshment,
    h::Hamiltonian,
    z::PhasePoint,
) = phasepoint(h, z.θ, ref.α * z.r + (1 - ref.α^2) * rand(rng, h.metric))
