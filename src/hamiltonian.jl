# TODO: add a type for kinetic energy

struct Hamiltonian{T<:Real,M<:AbstractMetric{T},F1,F2,A<:AbstractVector{T}}
    metric      ::  M
    logπ        ::  F1
    ∂logπ∂θ     ::  F2
    # Below are for efficient memory allocation
    _∂H∂θ       ::  A
    _∂H∂r       ::  A
    _r          ::  A
end

function Hamiltonian(metric::M, logπ::F1, ∂logπ∂θ::F2) where {T<:Real,M<:AbstractMetric{T},F1,F2}
    return Hamiltonian(metric, logπ, ∂logπ∂θ, zeros(T, metric.dim), zeros(T, metric.dim), zeros(T, metric.dim))
end

function ∂H∂θ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    h._∂H∂θ .= -h.∂logπ∂θ(θ)
    return h._∂H∂θ
end

function ∂H∂r(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._∂H∂r .= r
    return h._∂H∂r
end

function ∂H∂r(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._∂H∂r .= h.metric.M⁻¹ .* r
    return h._∂H∂r
end

function ∂H∂r(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    mul!(h._∂H∂r, h.metric.M⁻¹, r)
    return h._∂H∂r
end

function H(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return K(h, r, θ) + V(h, θ)
end

function V(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h.logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function K(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return sum(abs2, r) / 2
end

function K(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    return sum(abs2(r[i]) * h.metric.M⁻¹[i] for i in 1:length(r)) / 2
end

function K(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    mul!(h._∂H∂r, h.metric.M⁻¹, r)
    return dot(r, h._∂H∂r) / 2
end

# TODO: make sure the re-use of allocation doesn't caues problem
# Momentum sampler
function rand_momentum(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    return h._r
end

function rand_momentum(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    h._r .= h._r ./ h.metric.sqrtM⁻¹
    return h._r
end

function rand_momentum(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    ldiv!(h.metric.cholM⁻¹, h._r)
    return h._r
end
