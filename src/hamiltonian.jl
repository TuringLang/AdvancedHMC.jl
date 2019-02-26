# TODO: add a type for kinetic energy

struct Hamiltonian{T<:Real,M<:AbstractMetric{T},F1,F2,A<:AbstractVector{T}}
    metric      ::  M
    logπ        ::  F1
    dlogπdθ     ::  F2
    # Below are for efficient memory allocation
    _dHdθ       ::  A
    _dHdr       ::  A
    _r          ::  A
end

function Hamiltonian(metric::M, logπ::F1, dlogπdθ::F2) where {T<:Real,M<:AbstractMetric{T},F1,F2}
    return Hamiltonian(metric, logπ, dlogπdθ, zeros(T, metric.dim), zeros(T, metric.dim), zeros(T, metric.dim))
end

function dHdθ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    h._dHdθ .= -h.dlogπdθ(θ)
    return h._dHdθ
end

function dHdr(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    h._dHdr .= r
    return h._dHdr
end

function dHdr(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._dHdr .= h.metric.M⁻¹ .* r
    return h._dHdr
end

function dHdr(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}, r::AbstractVector{T}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    mul!(h._dHdr, h.metric.M⁻¹, r)
    return h._dHdr
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
    mul!(h._dHdr, h.metric.M⁻¹, r)
    return dot(r, h._dHdr) / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{T,UnitEuclideanMetric{T},F1,F2,A}) where {T<:Real,F1,F2,A<:AbstractVector{T}}
    return randn(h.metric.dim)
end

function rand_momentum(h::Hamiltonian{T,DiagEuclideanMetric{T,M},F1,F2,A}) where {T<:Real,M<:AbstractVector{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    return h._r ./ h.metric.sqrtM⁻¹
end

function rand_momentum(h::Hamiltonian{T,DenseEuclideanMetric{T,M},F1,F2,A}) where {T<:Real,M<:AbstractMatrix{T},F1,F2,A<:AbstractVector{T}}
    h._r .= randn.()
    return h.metric.cholM⁻¹ \ h._r
end
