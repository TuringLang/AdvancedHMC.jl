struct Hamiltonian{M<:AbstractMetric}
    metric      ::  M
    _logπ       ::  Function
    _dlogπdθ    ::  Function
end

# TODO: implement a helper function for those only _logπ is provided and use AD to provide _dlogπdθ

function _dHdθ(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return h._dlogπdθ(θ)
end

function _dHdr(h::Hamiltonian{UnitMetric}, r::AbstractVector{T}) where {T<:Real}
    return r
end

function _dHdr(h::Hamiltonian{DiagMetric{T}}, r::AbstractVector{T}) where {T<:Real}
    return h.metric.M⁻¹ .* r
end

function _dHdr(h::Hamiltonian{DenseMetric{T}}, r::AbstractVector{T}) where {T<:Real}
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
function _K(h::Hamiltonian{UnitMetric}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return sum(abs2, r) / 2
end

function _K(h::Hamiltonian{DiagMetric{T}}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return sum(abs2, r .* sqrt.(h.metric.M⁻¹)) / 2
end

function _K(h::Hamiltonian{DenseMetric{T}}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return p' * h.metric.M⁻¹ * p / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{UnitMetric}, θ::AbstractVector{T}) where {T<:Real}
    return randn(length(θ))
end

function rand_momentum(h::Hamiltonian{DiagMetric{T}}, θ::AbstractVector{T}) where {T<:Real}
    return randn(length(θ)) ./ sqrt.(h.metric.M⁻¹)
end

function rand_momentum(h::Hamiltonian{DenseMetric{T}}, θ::AbstractVector{T}) where {T<:Real}
    C = cholesky(Symmetric(h.metric.M⁻¹))
    return C.U \ randn(length(θ))
end
