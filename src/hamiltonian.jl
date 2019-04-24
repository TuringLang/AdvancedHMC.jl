# TODO: add a type for kinetic energy

# The constructor of `DualValue` will check numerical errors in
#   `value` and `gradient`.  That is `is_valid` will be performed automatically.
struct DualValue{Tv<:AbstractFloat, Tg<:AbstractVector{Tv}} <: AbstractFloat
    value::Tv    # Cached value, e.g. logπ(θ).
    gradient::Tg # Cached gradient, e.g. ∇logπ(θ).
end

struct DualFunction{Tf<:Function}
    f::Tf
    ∇f::Tf
end

# TODO: replace logπ and logκ with π, κ
# The constructor of `PhasePoint` will check numerical errors in
#   `θ`, `r` and `h`. That is `is_valid` will be performed automatically.
struct PhasePoint{T<:AbstractVector, V<:AbstractFloat}
    θ::T # position variables / parameters
    r::T # momentum variables
    logπ::V # cached potential energy for the current θ
    logκ::V # cached kinect energy for the current r
    function PhasePoint(θ::T, r::T, ℓπ::V, ℓκ::V) where {T,V}
        @argcheck length(θ) == length(r) == length(ℓπ.gradient) == length(ℓπ.gradient)
        if !(all(isfinite, θ) && all(isfinite, r) && all(isfinite, ℓπ) && all(isfinite, ℓκ))
            @warn "Numerical error has been detected. Rejecting current proposal..."
            ℓκ = DualValue(-Inf, ℓκ.gradient)
            ℓπ = DualValue(-Inf, ℓπ.gradient)
        end
        new{T,V}(θ, r, ℓπ, ℓκ)
    end
end

Base.isfinite(v::DualValue) = all(isfinite(v.value)) && all(isfinite(v.gradient))

struct Hamiltonian{M<:AbstractMetric, Tlogπ, T∂logπ∂θ}
    metric::M
    logπ::Tlogπ
    # The following will be merged into logπ::DualFunction
    ∂logπ∂θ::T∂logπ∂θ
end

# Create a `Hamiltonian` with a new `M⁻¹`
(h::Hamiltonian)(M⁻¹) = Hamiltonian(h.metric(M⁻¹), h.logπ, h.∂logπ∂θ)

# TODO: rename to ∇π and ∇κ
∂H∂θ(h::Hamiltonian, θ::AbstractVector) = -h.∂logπ∂θ(θ)

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVector) = copy(r)
∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ .* r
∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ * r


# TODO: add cache for gradients
function phasepoint(h::Hamiltonian, θ::AbstractVector, r::AbstractVector)
    π = DualValue(-kinetic_energy(h, r, θ), ∂H∂θ(h, θ))
    κ = DualValue(-potential_energy(h, θ), ∂H∂r(h, r))
    return PhasePoint(θ, r, π, κ)
end

neg_energy(z::PhasePoint) = - z.logπ.value - z.logκ.value
rand_momentum(
    rng::AbstractRNG,
    z::PhasePoint,
    h::Hamiltonian
) = phasepoint(h, z.θ, rand_momentum(rng, h))
rand_momentum(z::PhasePoint, h::Hamiltonian) = phasepoint(h, z.θ, rand_momentum(h))

function hamiltonian_energy(h::Hamiltonian, θ::AbstractVector, r::AbstractVector)
    K = kinetic_energy(h, r, θ)
    if isnan(K)
        K = Inf
        @warn "Kinetic energy is `NaN` and is set to `Inf`."
    end
    V = potential_energy(h, θ)
    if isnan(V)
        V = Inf
        @warn "Potential energy is `NaN` and is set to `Inf`."
    end
    return K + V
end

potential_energy(h::Hamiltonian, θ::AbstractVector) = -h.logπ(θ)

# Kinetic energy
# NOTE: the general form (i.e. non-Euclidean) of K depends on both θ and r
kinetic_energy(h::Hamiltonian{<:UnitEuclideanMetric}, r, θ) = sum(abs2, r) / 2
function kinetic_energy(h::Hamiltonian{<:DiagEuclideanMetric}, r, θ)
    return sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)) / 2
end
function kinetic_energy(h::Hamiltonian{<:DenseEuclideanMetric}, r, θ)
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return dot(r, h.metric._temp) / 2
end

# Momentum sampler
function rand_momentum(rng::AbstractRNG, h::Hamiltonian{<:UnitEuclideanMetric})
    return randn(rng, h.metric.dim)
end
function rand_momentum(rng::AbstractRNG, h::Hamiltonian{<:DiagEuclideanMetric})
    r = randn(rng, h.metric.dim)
    r ./= h.metric.sqrtM⁻¹
    return r
end
function rand_momentum(rng::AbstractRNG, h::Hamiltonian{<:DenseEuclideanMetric})
    r = randn(rng, h.metric.dim)
    ldiv!(h.metric.cholM⁻¹, r)
    return r
end

# TODO: re-write this function such that it uses a package level global
#  RNG (e.g. Advanced.RNG) instead of julia run-time level RNG
rand_momentum(h::Hamiltonian) = rand_momentum(GLOBAL_RNG, h)
