# TODO: add a type for kinetic energy

struct Hamiltonian{T<:Real,M<:AbstractMetric{T},F1,F2,A<:AbstractVector{T}}
    metric      ::  M
    _logπ       ::  F1
    _dlogπdθ    ::  F2
    # Below are for efficient memory allocation
    _dHdθ       ::  A
    _dHdr       ::  A
    _r          ::  A
end

function Hamiltonian(metric::M, _logπ::F1, _dlogπdθ::F2) where {T<:Real,M<:AbstractMetric{T},F1,F2}
    return Hamiltonian(metric, _logπ, _dlogπdθ, zeros(T, metric.dim), zeros(T, metric.dim), zeros(T, metric.dim))
end

function _dHdθ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    h._dHdθ .= -h._dlogπdθ(θ)
    return h._dHdθ
end

function _dHdr(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._dHdr .= r
    return h._dHdr
end

function _dHdr(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._dHdr .= h.metric.M⁻¹ .* r
    return h._dHdr
end

function _dHdr(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    mul!(h._dHdr, h.metric.M⁻¹, r)
    return h._dHdr
end

function _H(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return _K(h, r, θ) + _V(h, θ)
end

function _V(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h._logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function _K(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return sum(abs2, r) / 2
end

function _K(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    t1 = BroadcastArray(abs2, r)
    t2 = BroadcastArray(*, t1, h.metric.M⁻¹)
    return sum(t2) / 2
end

function _K(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    return r' * h.metric.M⁻¹ * r / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, θ::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    return h._r
end

function rand_momentum(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, θ::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    h._r .= h._r ./ h.metric.sqrtM⁻¹
    return h._r
end

function rand_momentum(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, θ::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    h._r .= h.metric.cholM⁻¹ \ h._r
    return h._r
end
