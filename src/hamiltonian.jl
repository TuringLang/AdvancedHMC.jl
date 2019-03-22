# TODO: add a type for kinetic energy

struct Hamiltonian{T<:Real,M<:AbstractMetric{T},F1,F2}
    metric      ::  M
    logπ        ::  F1
    ∂logπ∂θ     ::  F2
end

function ∂H∂θ(h::Hamiltonian, θ::AV)::AV where {T<:Real,AV<:AbstractVector{T}}
    return -h.∂logπ∂θ(θ)
end

function ∂H∂r(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2}, r::A) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return copy(r)
end

function ∂H∂r(h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2}, r::A) where {T<:Real,A<:AbstractVector{T},F1,F2}
    return h.metric.M⁻¹ .* r
end

function ∂H∂r(h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2}, r::AV) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    return h.metric.M⁻¹ * r
end

function hamiltonian_energy(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return kinetic_energy(h, r, θ) + potential_energy(h, θ)
end

function potential_energy(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h.logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function kinetic_energy(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2}, r::A, ::A) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return sum(abs2, r) / 2
end

function kinetic_energy(h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2}, r::A, ::A) where {T<:Real,A<:AbstractVector{T},F1,F2}
    return sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)) / 2
end

function kinetic_energy(h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2}, r::AV, ::AV) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    mul!(h.metric._temp, h.metric.M⁻¹, r)
    return dot(r, h.metric._temp) / 2
end

# Momentum sampler
function rand_momentum(rng::AbstractRNG, h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return randn(rng, h.metric.dim)
end

function rand_momentum(rng::AbstractRNG, h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2}) where {T<:Real,A<:AbstractVector{T},F1,F2}
    h.metric._temp .= randn.(Ref(rng))
    return h.metric._temp ./ h.metric.sqrtM⁻¹
end

function rand_momentum(rng::AbstractRNG, h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2}) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    h.metric._temp .= randn.(Ref(rng))
    return h.metric.cholM⁻¹ \ h.metric._temp
end

rand_momentum(h::Hamiltonian) = rand_momentum(GLOBAL_RNG, h)
