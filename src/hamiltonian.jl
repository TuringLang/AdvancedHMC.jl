# TODO: add a type for kinetic energy

# The constructor of `DualValue` will check numerical errors in
#   `value` and `gradient`.  That is `is_valid` will be performed automatically.
struct DualValue{Tv<:AbstractFloat, Tg<:AbstractVector{Tv}}
    value::Tv    # Cached value, e.g. logπ(θ).
    gradient::Tg # Cached gradient, e.g. ∇logπ(θ).
end

struct DualFunction{Tf<:Function}
    f::Tf
    ∇f::Tf
end

# The constructor of `PhasePoint` will check numerical errors in
#   `θ`, `r` and `h`. That is `is_valid` will be performed automatically.
struct PhasePoint{T<:AbstractVector, Th<:DualValue}
    θ::T # position variables / parameters
    r::T # momentum variables
    logπ::Th # cached potential energy for the current θ
    logκ::Th # cached kinect energy for the current r
end

struct Hamiltonian{M<:AbstractMetric, Tlogπ, T∂logπ∂θ}
    metric::M
    logπ::Tlogπ
    # The following will be merged into logπ::DualFunction
    ∂logπ∂θ::T∂logπ∂θ
end

# Create a `Hamiltonian` with a new `M⁻¹`
(h::Hamiltonian)(M⁻¹) = Hamiltonian(h.metric(M⁻¹), h.logπ, h.∂logπ∂θ)

∂H∂θ(h::Hamiltonian, θ::AbstractVector) = -h.∂logπ∂θ(θ)

∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric}, r::AbstractVector) = copy(r)
∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ .* r
∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric}, r::AbstractVector) = h.metric.M⁻¹ * r

function function hamiltonian_energy(h::Hamiltonian, θ::AbstractVector, r::AbstractVector)
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

##
## API: required by Turing.Gibbs
##

# function step(rng::AbstractRNG, h::Hamiltonian, prop::AbstractTrajectory{I}, θ::AbstractVector{T}) where {T<:Real,I<:AbstractIntegrator}
#     r = rand_momentum(rng, h)
#     θ_new, r_new, α, H_new = transition(rng, prop, h, θ, r)
#     return θ_new, H_new, α
# end
#
# step(h::Hamiltonian, p::AbstractTrajectory, θ::AbstractVector{T}) where {T<:Real} = step(GLOBAL_RNG, h, p, θ)
