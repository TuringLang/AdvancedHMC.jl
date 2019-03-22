abstract type AbstractMetric{T} end

struct UnitEuclideanMetric{T<:Real} <: AbstractMetric{T}
    dim :: Int
end

function UnitEuclideanMetric(θ::A) where {T<:Real,A<:AbstractVector{T}}
    return UnitEuclideanMetric{T}(length(θ))
end

struct DiagEuclideanMetric{T<:Real,A<:AbstractVector{T}} <: AbstractMetric{T}
    dim     :: Int
    # Diagnal of the inverse of the mass matrix
    M⁻¹     ::  A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹ ::  A
    # Pre-allocation for intermediate variables
    _temp   ::  A
end

function (::DiagEuclideanMetric)(M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    return DiagEuclideanMetric(M⁻¹)
end

function DiagEuclideanMetric(θ::A, M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    @assert length(θ) == length(M⁻¹)
    return DiagEuclideanMetric(M⁻¹)
end

function DiagEuclideanMetric(M⁻¹::V) where {T<:Real,V<:AbstractVector{T}}
    dim = length(M⁻¹)
    return DiagEuclideanMetric(dim, M⁻¹, sqrt.(M⁻¹), zeros(T, dim))
end

struct DenseEuclideanMetric{T<:Real,AV<:AbstractVector{T},AM<:AbstractMatrix{T}} <: AbstractMetric{T}
    dim     :: Int
    # Inverse of the mass matrix
    M⁻¹     ::  AM
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹ ::  UpperTriangular{T,AM}
    # Pre-allocation for intermediate variables
    _temp   ::  AV
end

function (::DenseEuclideanMetric)(M⁻¹::A) where {T<:Real,A<:AbstractMatrix{T}}
    return DenseEuclideanMetric(M⁻¹)
end

function DenseEuclideanMetric(θ::AV, M⁻¹::AM) where {T<:Real,AV<:AbstractVector{T},AM<:AbstractMatrix{T}}
    @assert length(θ) == size(M⁻¹, 1)
    return DenseEuclideanMetric(M⁻¹)
end

function DenseEuclideanMetric(M⁻¹::M) where {T<:Real,M<:AbstractMatrix{T}}
    dim = size(M⁻¹, 1)
    return DenseEuclideanMetric(dim, M⁻¹, cholesky(Symmetric(M⁻¹)).U, zeros(T, dim))
end
