abstract type AbstractMetric end

struct UnitEuclideanMetric <: AbstractMetric
    dim::Int
end

# Create a `UnitEuclideanMetric`; required for an unified interface
(ue::UnitEuclideanMetric)(::Nothing) = UnitEuclideanMetric(ue.dim)

function _string_diag(d, n_chars::Int=32) :: String
    s_diag = string(d)
    l = length(s_diag)
    s_dots = " ..."
    n_diag_chars = n_chars - length(s_dots)
    return s_diag[1:min(n_diag_chars,end)] * (l > n_diag_chars ? s_dots : "")
end

Base.show(io::IO, uem::UnitEuclideanMetric) = print(io, _string_diag(ones(uem.dim)))

struct DiagEuclideanMetric{A<:AbstractVector} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹     ::  A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹ ::  A
    # Pre-allocation for intermediate variables
    _temp   ::  A
end

function DiagEuclideanMetric(M⁻¹::AbstractVector{T}) where {T<:Real}
    return DiagEuclideanMetric(M⁻¹, sqrt.(M⁻¹), Vector{T}(undef, size(M⁻¹, 1)))
end
DiagEuclideanMetric(D::Int) = DiagEuclideanMetric(ones(Float64, D))

# Create a `DiagEuclideanMetric` with a new `M⁻¹`
(dem::DiagEuclideanMetric)(M⁻¹::AbstractVector{<:Real}) = DiagEuclideanMetric(M⁻¹)

Base.show(io::IO, dem::DiagEuclideanMetric) = print(io, _string_diag(dem.M⁻¹))

function Base.getproperty(dem::DiagEuclideanMetric, d::Symbol)
    return d === :dim ? size(getfield(dem, :M⁻¹), 1) : getfield(dem, d)
end


struct DenseEuclideanMetric{
    AV<:AbstractVector,
    AM<:AbstractMatrix,
    TcholM⁻¹<:UpperTriangular,
} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹::AM
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹::TcholM⁻¹
    # Pre-allocation for intermediate variables
    _temp::AV
end

function DenseEuclideanMetric(M⁻¹::AbstractMatrix{T}) where {T<:Real}
    _temp = Vector{T}(undef, size(M⁻¹, 1))
    return DenseEuclideanMetric(M⁻¹, cholesky(Symmetric(M⁻¹)).U, _temp)
end
DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Matrix{Float64}(I, D, D))

# Create a `DenseEuclideanMetric` with a new `M⁻¹`
(dem::DenseEuclideanMetric)(M⁻¹::AbstractMatrix{<:Real}) = DenseEuclideanMetric(M⁻¹)

Base.show(io::IO, dem::DenseEuclideanMetric) = print(io, _string_diag(diag(dem.M⁻¹)))

function Base.getproperty(dem::DenseEuclideanMetric, d::Symbol)
    return d === :dim ? size(getfield(dem, :M⁻¹), 1) : getfield(dem, d)
end
