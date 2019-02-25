abstract type AbstractMetric{T} end

struct UnitEuclideanMetric{T<:Real} <: AbstractMetric{T}
    dim :: Integer
end

function UnitEuclideanMetric(θ::A) where {T<:Real,A<:AbstractVector{T}}
    return UnitEuclideanMetric{T}(length(θ))
end

struct DiagEuclideanMetric{T<:Real,A<:AbstractVector{T}} <: AbstractMetric{T}
    dim     :: Integer
    # Diagnal of the inverse of the mass matrix
    M⁻¹     ::  A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹ ::  A
end

function DiagEuclideanMetric(θ::A, M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    @assert length(θ) == length(M⁻¹)
    return DiagEuclideanMetric(length(θ), M⁻¹, sqrt.(M⁻¹))
end

struct DenseEuclideanMetric{T<:Real,A<:AbstractMatrix{T}} <: AbstractMetric{T}
    dim     :: Integer
    # Inverse of the mass matrix
    M⁻¹     ::  A
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹ ::  UpperTriangular{T,A}
end

function DenseEuclideanMetric(θ::A1, M⁻¹::A2) where {T<:Real,A1<:AbstractVector{T},A2<:AbstractMatrix{T}}
    @assert length(θ) == size(M⁻¹, 1)
    return DenseEuclideanMetric(length(θ), M⁻¹, cholesky(Symmetric(M⁻¹)).U)
end
