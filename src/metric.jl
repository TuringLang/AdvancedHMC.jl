abstract type AbstractMetric end

struct UnitEuclideanMetric <: AbstractMetric end

struct DiagEuclideanMetric{A<:AbstractVector{<:Real}} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹ ::  A
end

struct DenseEuclideanMetric{A<:AbstractMatrix{<:Real}} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹ ::  A
end
