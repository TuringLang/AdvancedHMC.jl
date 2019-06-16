# TODO: add a type for kinetic energy
# TODO: add cache for gradients by letting ∂logπ∂θ return both log density and gradient

struct Hamiltonian{M<:AbstractMetric, Tlogπ, TF∂logπ∂θ, T∂H∂θ, T∂H∂r}
    metric::M
    ℓπ::Tlogπ
    ∂ℓπ∂θ!::TF∂logπ∂θ
    ∂H∂θ::T∂H∂θ
    ∂H∂r::T∂H∂r
end
Hamiltonian(metric::AbstractMetric, ℓπ, ∂ℓπ∂θ!) = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ!, zeros(length(metric)), zeros(length(metric)))

# Create a `Hamiltonian` with a new `M⁻¹`
(h::Hamiltonian)(M⁻¹) = Hamiltonian(h.metric(M⁻¹), h.ℓπ, h.∂ℓπ∂θ!, h.∂H∂θ, h.∂H∂r)

∂H∂θ!(h::Hamiltonian, θ::AbstractVector) = rmul!(h.∂ℓπ∂θ!(h.∂H∂θ, θ), -1)

∂H∂r!(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVector) = (h.∂H∂r .= r; h.∂H∂r)
∂H∂r!(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVector) = (h.∂H∂r .= h.metric.M⁻¹ .* r; h.∂H∂r)
∂H∂r!(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVector) = (mul!(h.∂H∂r, h.metric.M⁻¹, r); h.∂H∂r)

struct DualValue{Tv<:AbstractFloat, Tg<:AbstractVector{Tv}}
    value::Tv    # Cached value, e.g. logπ(θ).
    gradient::Tg # Cached gradient, e.g. ∇logπ(θ).
end

struct PhasePoint{T<:AbstractVector, V<:DualValue}
    θ::T  # Position variables / model parameters.
    r::T  # Momentum variables
    ℓπ::V # Cached neg potential energy for the current θ.
    ℓκ::V # Cached neg kinect energy for the current r.
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
    ℓπ = DualValue(neg_energy(h, r, θ), ∂H∂θ!(h, θ)),
    ℓκ = DualValue(neg_energy(h, θ), ∂H∂r!(h, r))
) where {T<:AbstractVector} = PhasePoint(θ, r, ℓπ, ℓκ)


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
    return -sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r))/2
end

function neg_energy(
    h::Hamiltonian{<:DenseEuclideanMetric},
    r::T,
    θ::T
) where {T<:AbstractVector}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return -dot(r, h.metric._temp) / 2
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
