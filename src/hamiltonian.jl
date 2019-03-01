# TODO: add a type for kinetic energy

struct Hamiltonian{T<:Real,M<:AbstractMetric{T},F1,F2,A<:AbstractVector{T}}
    metric      ::  M
    logπ        ::  F1
    ∂logπ∂θ     ::  F2
    # Below are for efficient memory allocation
    _∂H∂θ       ::  A
    _∂H∂r       ::  A
end

function Hamiltonian(metric::M, logπ::F1, ∂logπ∂θ::F2) where {T<:Real,M<:AbstractMetric{T},F1,F2}
    return Hamiltonian(metric, logπ, ∂logπ∂θ, zeros(T, metric.dim), zeros(T, metric.dim))
end

# TODO: make sure the re-use of allocation doesn't cause any problem
function ∂H∂θ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    h._∂H∂θ .= -h.∂logπ∂θ(θ)
    return h._∂H∂θ
end

function ∂H∂r(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::A) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._∂H∂r .= r
    return h._∂H∂r
end

function ∂H∂r(h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2,A}, r::A) where {T<:Real,A<:AbstractVector{T},F1,F2}
    h._∂H∂r .= h.metric.M⁻¹ .* r
    return h._∂H∂r
end

function ∂H∂r(h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2,AV}, r::AV) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    mul!(h._∂H∂r, h.metric.M⁻¹, r)
    return h._∂H∂r
end

function hamiltonian_energy(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return kinetic_energy(h, r, θ) + potential_energy(h, θ)
end

function potential_energy(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h.logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function kinetic_energy(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::A, ::A) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return sum(abs2, r) / 2
end

function kinetic_energy(h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2,A}, r::A, ::A) where {T<:Real,A<:AbstractVector{T},F1,F2}
    return sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)) / 2
end

function kinetic_energy(h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2,AV}, r::AV, ::AV) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    mul!(h._∂H∂r, h.metric.M⁻¹, r)
    return dot(r, h._∂H∂r) / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return randn(h.metric.dim)
end

function rand_momentum(h::Hamiltonian{T,DiagEuclideanMetric{T,A},F1,F2,A}) where {T<:Real,A<:AbstractVector{T},F1,F2}
    h.metric._r .= randn.()
    return h.metric._r ./ h.metric.sqrtM⁻¹
end

function rand_momentum(h::Hamiltonian{T,DenseEuclideanMetric{T,AV,AM},F1,F2,AV}) where {T<:Real,AM<:AbstractMatrix{T},F1,F2,AV<:AbstractVector{T}}
    h.metric._r .= randn.()
    return h.metric.cholM⁻¹ \ h.metric._r
end
