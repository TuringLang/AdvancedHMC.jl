# TODO: add a type for kinetic energy

struct Hamiltonian{M<:AbstractMetric,F1,F2}
    metric      ::  M
    _logπ       ::  F1
    _dlogπdθ    ::  F2
end

function Hamiltonian(metric::M, _logπ::F1, _dlogπdθ::F2) where {M<:AbstractMetric,F1,F2}
    return Hamiltonian{M,F1,F2}(metric, _logπ, _dlogπdθ)
end

function _dHdθ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h._dlogπdθ(θ)
end

function _dHdr(h::Hamiltonian{UnitEuclideanMetric{T},F1,F2}, r::AbstractVector{T}) where {T<:Real,F1,F2}
    return r
end

function _dHdr(h::Hamiltonian{DiagEuclideanMetric{T,M},F1,F2}, r::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2}
    return h.metric.M⁻¹ .* r
end

function _dHdr(h::Hamiltonian{DenseEuclideanMetric{T,M},F1,F2}, r::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2}
    return h.metric.M⁻¹ * r
end

function _H(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return _K(h, r, θ) + _V(h, θ)
end

function _V(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h._logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function _K(h::Hamiltonian{UnitEuclideanMetric{T},F1,F2}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,F1,F2}
    return sum(abs2, r) / 2
end

function _K(h::Hamiltonian{DiagEuclideanMetric{T,M},F1,F2}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2}
    t1 = BroadcastArray(abs2, r)
    t2 = BroadcastArray(*, t1, h.metric.M⁻¹)
    return sum(t2) / 2
end

function _K(h::Hamiltonian{DenseEuclideanMetric{T,M},F1,F2}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2}
    return r' * h.metric.M⁻¹ * r / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{UnitEuclideanMetric{T},F1,F2}, θ::AbstractVector{T}) where {T<:Real,F1,F2}
    return randn(length(θ))
end

function rand_momentum(h::Hamiltonian{DiagEuclideanMetric{T,M},F1,F2}, θ::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2}
    return randn(length(θ)) ./ h.metric.sqrtM⁻¹
end

function rand_momentum(h::Hamiltonian{DenseEuclideanMetric{T,M},F1,F2}, θ::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2}
    return h.metric.cholM⁻¹ \ randn(length(θ))
end
