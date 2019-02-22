struct Hamiltonian{T<:Real,M<:AbstractMetric}
    metric  ::  M{T}
    _logπ    ::  Function
    _dlogπdθ ::  Function
end

function _dHdθ{T}(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return h._dlogπdθ(θ)
end

function _dHdr{T}(h::Hamiltonian{T,UnitMetric}, r::AbstractVector{T}) where {T<:Real}
    return r
end

function _dHdr{T}(h::Hamiltonian{T,DiagMetric}, r::AbstractVector{T}) where {T<:Real}
    return h.metric.M⁻¹ .* r
end

function _dHdr{T}(h::Hamiltonian{T,DenseMetric}, r::AbstractVector{T}) where {T<:Real}
    return h.metric.M⁻¹ * r
end

function _H{T}(h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return _K(r, θ) + _V(θ)
end

function _V{T}(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return -h._logπ(θ)
end

# Kinetic energy
# NOTE: the general form of K depends on both θ and r
function _K{T}(h::Hamiltonian{T,UnitMetric}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return sum(abs2, r) / 2
end

function _K{T}(h::Hamiltonian{T,DiagMetric}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return sum(abs2, r .* sqrt.(h.metric.M⁻¹)) / 2
end

function _K{T}(h::Hamiltonian{T,DenseMetric}, r::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
    return p' * h.metric.M⁻¹ * p / 2
end

# Momentum sampler
function rand_momentum(h::Hamiltonian{T,UnitMetric}, θ::AbstractVector{T})
    return randn(length(θ))
end

function rand_momentum(h::Hamiltonian{T,DiagMetric}, θ::AbstractVector{T})
    return randn(length(θ)) ./ sqrt.(h.metric.M⁻¹)
end

function rand_momentum(h::Hamiltonian{T,DenseMetric}, θ::AbstractVector{T})
    C = cholesky(Symmetric(h.metric.M⁻¹))
    return C.U \ randn(length(θ))
end
