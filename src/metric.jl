abstract type AbstractMetric{T} end

struct UnitEuclideanMetric{T<:Real} <: AbstractMetric{T}
    dim :: Int
end

function _string_diag(d, n_chars::Int=32) :: String
    s_diag = string(d)
    l = length(s_diag)
    s_dots = " ..."
    n_diag_chars = n_chars - length(s_dots)
    return s_diag[1:min(n_diag_chars,end)] * (l > n_diag_chars ? s_dots : "")
end
Base.show(io::IO, uem::UnitEuclideanMetric{T}) where {T<:Real} = print(io, _string_diag(ones(T, uem.dim)))

function UnitEuclideanMetric(θ::A) where {T<:Real,A<:AbstractVector{T}}
    return UnitEuclideanMetric{T}(length(θ))
end

# Create a `UnitEuclideanMetric`; required for an unified interface
function (ue::UnitEuclideanMetric{T})(::Nothing) where {T<:Real}
    return UnitEuclideanMetric{T}(ue.dim)
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

Base.show(io::IO, dem::DiagEuclideanMetric) = print(io, _string_diag(dem.M⁻¹))

# Create a `DiagEuclideanMetric` with a new `M⁻¹`
function (dem::DiagEuclideanMetric)(M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    return DiagEuclideanMetric(dem.dim, M⁻¹)
end

function DiagEuclideanMetric(θ::A) where {T<:Real,A<:AbstractVector{T}}
    dim = length(θ)
    return DiagEuclideanMetric(dim, ones(T, dim))
end

function DiagEuclideanMetric(θ::A, M⁻¹::A) where {T<:Real,A<:AbstractVector{T}}
    return DiagEuclideanMetric(length(θ), M⁻¹)
end

function DiagEuclideanMetric(dim::Int, M⁻¹::V) where {T<:Real,V<:AbstractVector{T}}
    @assert dim == length(M⁻¹)
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

Base.show(io::IO, dem::DenseEuclideanMetric) = print(io, _string_diag(diag(dem.M⁻¹)))

# Create a `DenseEuclideanMetric` with a new `M⁻¹`
function (dem::DenseEuclideanMetric)(M⁻¹::A) where {T<:Real,A<:AbstractMatrix{T}}
    return DenseEuclideanMetric(dem.dim, M⁻¹)
end

function DenseEuclideanMetric(θ::A) where {T<:Real,A<:AbstractVector{T}}
    dim = length(θ)
    return DenseEuclideanMetric(dim, Matrix{T}(I, dim, dim))
end

function DenseEuclideanMetric(θ::AV, M⁻¹::AM) where {T<:Real,AV<:AbstractVector{T},AM<:AbstractMatrix{T}}
    return DenseEuclideanMetric(length(θ), M⁻¹)
end

function DenseEuclideanMetric(dim::Int, M⁻¹::M) where {T<:Real,M<:AbstractMatrix{T}}
    @assert dim == size(M⁻¹, 1)
    return DenseEuclideanMetric(dim, M⁻¹, cholesky(Symmetric(M⁻¹)).U, zeros(T, dim))
end
