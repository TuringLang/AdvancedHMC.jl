# TODO: add a type for kinetic energy

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
# NOTE: the general form of K depends on both θ and r
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

rand_momentum(h::Hamiltonian) = rand_momentum(GLOBAL_RNG, h)
