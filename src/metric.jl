abstract type AbstractMetric end

struct UnitMetric <: AbstractMetric end

struct DiagMetric{A<:AbstractVector{<:Real}} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹ ::  A
end

struct DenseMetric{A<:AbstractMatrix{<:Real}} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹ ::  A
end
