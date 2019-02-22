abstract type AbstractMetric end

struct UnitMetric <: AbstractMetric end

struct DiagMetric{T<:Real} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹ ::  AbstractVector{T}
end

struct DenseMetric{T<:Real} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹ ::  AbstractMatrix{T}
end
