# TODO: add a type for kinetic energy
# TODO: add cache for gradients by letting ∂logπ∂θ return both log density and gradient

struct Hamiltonian{M<:AbstractMetric, Tlogπ, T∂logπ∂θ}
    metric::M
    ℓπ::Tlogπ
    ∂ℓπ∂θ::T∂logπ∂θ
end
Base.show(io::IO, h::Hamiltonian) = print(io, "Hamiltonian(metric=$(h.metric))")


struct DualValue{Tv<:AbstractFloat, Tg<:AbstractVector{Tv}}
    value::Tv    # Cached value, e.g. logπ(θ).
    gradient::Tg # Cached gradient, e.g. ∇logπ(θ).
end

# `∂H∂θ` now returns `(logprob, -∂ℓπ∂θ)`
function ∂H∂θ(h::Hamiltonian, θ::AbstractVector)
    res = h.∂ℓπ∂θ(θ)
    return DualValue(res[1], -res[2])
end

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVector) = copy(r)
∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ .* r
∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ * r

struct PhasePoint{T<:AbstractVector, V<:DualValue}
    θ::T  # Position variables / model parameters.
    r::T  # Momentum variables
    ℓπ::V # Cached neg potential energy for the current θ.
    ℓκ::V # Cached neg kinect energy for the current r.
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T,V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓπ.gradient)
        isfiniteθ, isfiniter, isfiniteℓπ, isfiniteℓκ = isfinite(θ), isfinite(r), isfinite(ℓπ), isfinite(ℓκ)
        if !(isfiniteθ && isfiniter && isfiniteℓπ && isfiniteℓκ)
            @warn "The current proposal will be rejected due to numerical error(s)." isfiniteθ isfiniter isfiniteℓπ isfiniteℓκ
            ℓπ = DualValue(-Inf, ℓπ.gradient)
            ℓκ = DualValue(-Inf, ℓκ.gradient)
        end
        new{T,V}(θ, r, ℓπ, ℓκ)
    end
end

phasepoint(
    h::Hamiltonian,
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r))
) where {T<:AbstractVector} = PhasePoint(θ, r, ℓπ, ℓκ)

# If position variable and momentum variable are in different containers,
# move the momentum variable to that of the position variable.
# This is needed for AHMC to work with CuArrays (without depending on it).
phasepoint(
    h::Hamiltonian,
    θ::T1,
    _r::T2;
    r=T1(_r),
    ℓπ=∂H∂θ(h, θ),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, r))
) where {T1<:AbstractVector,T2<:AbstractVector} = PhasePoint(θ, r, ℓπ, ℓκ)

Base.isfinite(v::DualValue) = all(isfinite, v.value) && all(isfinite, v.gradient)
Base.isfinite(v::AbstractVector) = all(isfinite, v)
Base.isfinite(z::PhasePoint) = isfinite(z.ℓπ) && isfinite(z.ℓκ)

###
### Negative energy (or log probability) functions.
### NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r.
###

neg_energy(z::PhasePoint) = z.ℓπ.value + z.ℓκ.value

neg_energy(h::Hamiltonian, θ::AbstractVector) = h.ℓπ(θ)

neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = -sum(abs2, r) / 2

function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector}
    _r = [abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)]
    return -sum(_r) / 2
end

function neg_energy(
    h::Hamiltonian{<:DenseEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return -dot(r, h.metric._temp) / 2
end

energy(args...) = -neg_energy(args...)

####
#### Momentum refreshment
####

refresh(
    rng::AbstractRNG,
    z::PhasePoint,
    h::Hamiltonian
) = phasepoint(h, z.θ, rand(rng, h.metric))
