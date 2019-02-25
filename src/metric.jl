abstract type AbstractMetric{T} end

struct UnitEuclideanMetric{T<:Real} <: AbstractMetric{T} end

struct DiagEuclideanMetric{T<:Real,A<:AbstractVector{T}} <: AbstractMetric{T}
    # Diagnal of the inverse of the mass matrix
    M⁻¹     ::  A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹ ::  A
end

function DiagEuclideanMetric(M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    return DiagEuclideanMetric{T,A}(M⁻¹, sqrt.(M⁻¹))
end

struct DenseEuclideanMetric{T<:Real,A<:AbstractMatrix{T}} <: AbstractMetric{T}
    # Inverse of the mass matrix
    M⁻¹     ::  A
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹ ::  UpperTriangular{T,A}
end

function DenseEuclideanMetric(M⁻¹::A) where {T<:Real,A<:AbstractMatrix{T}}
    return DenseEuclideanMetric{T,A}(M⁻¹, cholesky(Symmetric(M⁻¹)).U)
end
