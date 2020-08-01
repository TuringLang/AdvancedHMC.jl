abstract type AbstractRefreshment end

struct Kinetic{M<:AbstractMetric, R<:AbstractRefreshment}
    metric::M
    refreshment::R
end

Base.show(io::IO, k::Kinetic) = 
    print(io, "Kinetic(metric=$(k.metric), refreshment=$(k.refreshment))")

struct Hamiltonian{K<:Kinetic, Tlogπ, T∂logπ∂θ}
    kinetic::K
    ℓπ::Tlogπ
    ∂ℓπ∂θ::T∂logπ∂θ
end
Base.show(io::IO, h::Hamiltonian) = print(io, "Hamiltonian(kinetic=$(h.kinetic))")

# Support the old syntax by using full refreshment
Hamiltonian(m::AbstractMetric, args...) = Hamiltonian(Kinetic(m, FullRefreshment()), args...)

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

∂H∂r(k::Kinetic{<:UnitEuclideanMetric}, r::AbstractVecOrMat) = copy(r)
∂H∂r(k::Kinetic{<:DiagEuclideanMetric}, r::AbstractVecOrMat) = k.metric.M⁻¹ .* r
∂H∂r(k::Kinetic{<:DenseEuclideanMetric}, r::AbstractVecOrMat) = k.metric.M⁻¹ * r
∂H∂r(h::Hamiltonian, r) = ∂H∂r(h.kinetic, r)

struct PhasePoint{T<:AbstractVecOrMat{<:AbstractFloat}, V<:DualValue}
    θ::T  # Position variables / model parameters.
    r::T  # Momentum variables
    ℓπ::V # Cached neg potential energy for the current θ.
    ℓκ::V # Cached neg kinect energy for the current r.
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T, V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓπ.gradient)
        if any(isfinite.((θ, r, ℓπ, ℓκ)) .== false)
            @warn "The current proposal will be rejected due to numerical error(s)." isfinite.((θ, r, ℓπ, ℓκ))
            ℓπ = DualValue(map(v -> isfinite(v) ? v : -Inf, ℓπ.value), ℓπ.gradient)
            ℓκ = DualValue(map(v -> isfinite(v) ? v : -Inf, ℓκ.value), ℓκ.gradient)
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
    ℓκ=DualValue(neg_energy(h.kinetic, r, θ), ∂H∂r(h, r))
) where {T<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

# If position variable and momentum variable are in different containers,
# move the momentum variable to that of the position variable.
# This is needed for AHMC to work with CuArrays (without depending on it).
phasepoint(
    h::Hamiltonian,
    θ::T1,
    _r::T2;
    r=T1(_r),
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h.kinetic, r, θ), ∂H∂r(h, r))
) where {T1<:AbstractVecOrMat,T2<:AbstractVecOrMat} = PhasePoint(θ, r, ℓπ, ℓκ)

Base.isfinite(v::DualValue) = all(isfinite, v.value) && all(isfinite, v.gradient)
Base.isfinite(v::AbstractVecOrMat) = all(isfinite, v)
Base.isfinite(z::PhasePoint) = isfinite(z.ℓπ) && isfinite(z.ℓκ)

###
### Negative energy (or log probability) functions.
### NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r.
###

neg_energy(z::PhasePoint) = z.ℓπ.value + z.ℓκ.value

neg_energy(
    k::Kinetic{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = -sum(abs2, r) / 2

neg_energy(
    k::Kinetic{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractMatrix} = -vec(sum(abs2, r; dims=1)) / 2

neg_energy(
    k::Kinetic{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = -sum(abs2.(r) .* k.metric.M⁻¹) / 2

neg_energy(
    k::Kinetic{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractMatrix} = -vec(sum(abs2.(r) .* k.metric.M⁻¹; dims=1) ) / 2

function neg_energy(
    k::Kinetic{<:DenseEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVecOrMat}
    mul!(k.metric._temp, k.metric.M⁻¹, r)
    return -dot(r, k.metric._temp) / 2
end

energy(args...) = -neg_energy(args...)

# Helper function to initialise a phase point
phasepoint(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    θ::AbstractVecOrMat{T},
    h::Hamiltonian
) where {T<:Real} = phasepoint(h, θ, rand(rng, h.kinetic.metric))

########################
# Momentum refreshment #
########################

struct FullRefreshment <: AbstractRefreshment end

refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    z::PhasePoint,
    h::Hamiltonian{<:Kinetic{M, R}}
) where {M, R<:FullRefreshment} = phasepoint(h, z.θ, rand(rng, h.kinetic.metric))

struct PartialRefreshment{T<:AbstractFloat} <: AbstractRefreshment
    α::T
end

"""
Partial momentum refreshment.

Ref: Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov chain monte carlo 2.11 (2011): 2.
"""
function refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    z::PhasePoint,
    h::Hamiltonian{<:Kinetic{M, R}}
) where {M, R<:PartialRefreshment}
    @unpack α = h.kinetic.refreshment
    return phasepoint(h, z.θ, α * z.r + (1 - α^2) * rand(rng, h.kinetic.metric))
end
