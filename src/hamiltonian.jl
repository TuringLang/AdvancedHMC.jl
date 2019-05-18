# TODO: add a type for kinetic energy
# TODO: add cache for gradients by letting ∂logπ∂θ return both log density and gradient

struct Hamiltonian{M<:AbstractMetric, Tlogπ, T∂logπ∂θ}
    metric::M
    logπ::Tlogπ
    ∂logπ∂θ::T∂logπ∂θ
end

# Create a `Hamiltonian` with a new `M⁻¹`
(h::Hamiltonian)(M⁻¹) = Hamiltonian(h.metric(M⁻¹), h.logπ, h.∂logπ∂θ)

∂H∂θ(h::Hamiltonian, θ::AbstractVector) = -h.∂logπ∂θ(θ)

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVector) = copy(r)
∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ .* r
∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ * r


# The constructor of `DualValue` will check numerical errors in
#   `value` and `gradient`.  That is `is_valid` will be performed automatically.
struct DualValue{Tv<:AbstractFloat, Tg<:AbstractVector{Tv}}
    value::Tv    # Cached value, e.g. logπ(θ).
    gradient::Tg # Cached gradient, e.g. ∇logπ(θ).
end

# TODO: replace logπ and logκ with ℓπ, ℓκ??
# The constructor of `PhasePoint` will check numerical errors in
#   `θ`, `r` and `h`. That is `is_valid` will be performed automatically.
struct PhasePoint{T<:AbstractVector, V<:DualValue}
    θ::T # position variables / parameters
    r::T # momentum variables
    logπ::V # cached potential energy for the current θ
    logκ::V # cached kinect energy for the current r
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T,V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓπ.gradient)
        # if !(all(isfinite, θ) && all(isfinite, r) && all(isfinite, ℓπ) && all(isfinite, ℓκ))
        if !(isfinite(θ) && isfinite(r) && isfinite(ℓπ) && isfinite(ℓκ))
            @warn "The current proposal will be rejected (due to numerical error(s))..."
            ℓκ = DualValue(-Inf, ℓκ.gradient)
            ℓπ = DualValue(-Inf, ℓπ.gradient)
        end
        new{T,V}(θ, r, ℓπ, ℓκ)
    end
end

phasepoint(
    h::Hamiltonian,
    θ::T,
    r::T;
    π = DualValue(-kinetic_energy(h, r, θ), ∂H∂θ(h, θ)),
    κ = DualValue(-potential_energy(h, θ), ∂H∂r(h, r))
) where {T<:AbstractVector} = PhasePoint(θ, r, π, κ)


Base.isfinite(v::DualValue) = all(isfinite, v.value) && all(isfinite, v.gradient)
Base.isfinite(v::AbstractVector) = all(isfinite, v)
Base.isfinite(z::PhasePoint) = isfinite(z.logπ) && isfinite(z.logκ)

###
### Negative energy (or log probability) functions.
### NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r.
###

neg_energy(z::PhasePoint) = - z.logπ.value - z.logκ.value

potential_energy(h::Hamiltonian, θ::AbstractVector) = -h.logπ(θ)

kinetic_energy(
    h::Hamiltonian{<:UnitEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = sum(abs2, r) / 2

kinetic_energy(
    h::Hamiltonian{<:DiagEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector} = sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)) / 2

function kinetic_energy(
    h::Hamiltonian{<:DenseEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return dot(r, h.metric._temp) / 2
end

####
#### Momentum sampler
####

rand_momentum(
    rng::AbstractRNG,
    z::PhasePoint,
    h::Hamiltonian
) = phasepoint(h, z.θ, rand(rng, h.metric))

rand_momentum(
    z::PhasePoint,
    h::Hamiltonian
) = phasepoint(h, z.θ, rand(GLOBAL_RNG, h.metric))
