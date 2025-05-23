"""
$(TYPEDEF)

Abstract type for preconditioning metrics. 
"""
abstract type AbstractMetric end

_string_M⁻¹(mat::AbstractMatrix, n_chars::Int=32) = _string_M⁻¹(diag(mat), n_chars)
function _string_M⁻¹(vec::AbstractVector, n_chars::Int=32)
    s_vec = string(vec)
    l = length(s_vec)
    s_dots = " ...]"
    n_diag_chars = n_chars - length(s_dots)
    return s_vec[1:min(n_diag_chars, end)] * (l > n_diag_chars ? s_dots : "")
end

struct UnitEuclideanMetric{T,A<:Union{Tuple{Int},Tuple{Int,Int}}} <: AbstractMetric
    M⁻¹::UniformScaling{T}
    size::A
end

function UnitEuclideanMetric(::Type{T}, sz) where {T}
    return UnitEuclideanMetric(UniformScaling{T}(one(T)), sz)
end
UnitEuclideanMetric(sz) = UnitEuclideanMetric(Float64, sz)
function UnitEuclideanMetric(::Type{T}, dim::Int) where {T}
    return UnitEuclideanMetric(UniformScaling{T}(one(T)), (dim,))
end
UnitEuclideanMetric(dim::Int) = UnitEuclideanMetric(Float64, (dim,))

renew(ue::UnitEuclideanMetric, M⁻¹) = UnitEuclideanMetric(M⁻¹, ue.size)

Base.eltype(::UnitEuclideanMetric{T}) where {T} = T
Base.size(e::UnitEuclideanMetric) = e.size
Base.size(e::UnitEuclideanMetric, dim::Int) = e.size[dim]
function Base.show(io::IO, ::MIME"text/plain", uem::UnitEuclideanMetric{T}) where {T}
    return print(
        io,
        "UnitEuclideanMetric{$T} with size $(size(uem)) mass matrix:\n",
        _string_M⁻¹(ones(uem.size)),
    )
end

struct DiagEuclideanMetric{T,A<:AbstractVecOrMat{T}} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹::A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹::A
    # Pre-allocation for intermediate variables
    _temp::A
end

function DiagEuclideanMetric(M⁻¹::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    return DiagEuclideanMetric(M⁻¹, sqrt.(M⁻¹), similar(M⁻¹))
end
DiagEuclideanMetric(::Type{T}, sz) where {T} = DiagEuclideanMetric(ones(T, sz...))
DiagEuclideanMetric(sz) = DiagEuclideanMetric(Float64, sz)
DiagEuclideanMetric(::Type{T}, dim::Int) where {T} = DiagEuclideanMetric(ones(T, dim))
DiagEuclideanMetric(dim::Int) = DiagEuclideanMetric(Float64, dim)

renew(ue::DiagEuclideanMetric, M⁻¹) = DiagEuclideanMetric(M⁻¹)

Base.eltype(::DiagEuclideanMetric{T}) where {T} = T
Base.size(e::DiagEuclideanMetric, dim...) = size(e.M⁻¹, dim...)
function Base.show(io::IO, ::MIME"text/plain", dem::DiagEuclideanMetric{T}) where {T}
    return print(
        io,
        "DiagEuclideanMetric{$T} with size $(size(dem)) mass matrix:\n",
        _string_M⁻¹(dem.M⁻¹),
    )
end

struct DenseEuclideanMetric{
    T,
    AV<:AbstractVecOrMat{T},
    AM<:Union{AbstractMatrix{T},AbstractArray{T,3}},
    TcholM⁻¹<:UpperTriangular{T},
} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹::AM
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹::TcholM⁻¹
    # Pre-allocation for intermediate variables
    _temp::AV
end

# TODO: make dense mass matrix support matrix-mode parallel
function DenseEuclideanMetric(
    M⁻¹::Union{AbstractMatrix{T},AbstractArray{T,3}}
) where {T<:AbstractFloat}
    _temp = Vector{T}(undef, first(size(M⁻¹)))
    return DenseEuclideanMetric(M⁻¹, cholesky(Symmetric(M⁻¹)).U, _temp)
end
DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
function DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T}
    return DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
end
DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, sz)

renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

Base.eltype(::DenseEuclideanMetric{T}) where {T} = T
Base.size(e::DenseEuclideanMetric, dim...) = size(e._temp, dim...)
function Base.show(io::IO, ::MIME"text/plain", dem::DenseEuclideanMetric{T}) where {T}
    return print(
        io,
        "DenseEuclideanMetric{$T} with size $(size(dem)) mass matrix:\n",
        _string_M⁻¹(dem.M⁻¹),
    )
end

"""
    RankUpdateEuclideanMetric{T,AM,AB,AD,F} <: AbstractMetric

A Gaussian Euclidean metric whose inverse is constructed by rank-updates.

# Fields

$(TYPEDFIELDS)

# Constructors

    RankUpdateEuclideanMetric(n::Int)
    RankUpdateEuclideanMetric(M⁻¹, B, D)

 - Construct a Gaussian Euclidean metric of size `(n, n)` with `M⁻¹` being diagonal matrix.
 - Construct a Gaussian Euclidean metric of `M⁻¹`, where `M⁻¹` should be a full rank positive definite matrix,
    and `B` `D` must be chose so that the Woodbury matrix `W = M⁻¹ + B D B^\\mathrm{T}` is positive definite.

# Example

```julia
julia> RankUpdateEuclideanMetric(3)
RankUpdateEuclideanMetric(diag=[1.0, 1.0, 1.0])
```

# References

 - Ben Bales, Arya Pourzanjani, Aki Vehtari, Linda Petzold, Selecting the Metric in Hamiltonian Monte Carlo, 2019
"""
struct RankUpdateEuclideanMetric{T,AM<:AbstractVecOrMat{T},AB,AD,F} <: AbstractMetric
    "Diagnal of the inverse of the mass matrix"
    M⁻¹::AM
    B::AB
    D::AD
    factorization::F
end

function woodbury_factorize(A, B, D)
    cholA = cholesky(A isa Diagonal ? A : Symmetric(A))
    U = cholA.U
    Q, R = qr(U' \ B)
    V = cholesky(Symmetric(muladd(R, D * R', I))).U
    return (U=U, Q=Q, V=V)
end

function RankUpdateEuclideanMetric(n::Int)
    M⁻¹ = Diagonal(ones(n))
    B = zeros(n, 0)
    D = zeros(0, 0)
    factorization = woodbury_factorize(M⁻¹, B, D)
    return RankUpdateEuclideanMetric(M⁻¹, B, D, factorization)
end
function RankUpdateEuclideanMetric(::Type{T}, n::Int) where {T}
    M⁻¹ = Diagonal(ones(T, n))
    B = Matrix{T}(undef, n, 0)
    D = Matrix{T}(undef, 0, 0)
    factorization = woodbury_factorize(M⁻¹, B, D)
    return RankUpdateEuclideanMetric(M⁻¹, B, D, factorization)
end

function RankUpdateEuclideanMetric(M⁻¹, B, D)
    factorization = woodbury_factorize(M⁻¹, B, D)
    return RankUpdateEuclideanMetric(M⁻¹, B, D, factorization)
end

function RankUpdateEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T}
    return RankUpdateEuclideanMetric(T, first(sz))
end
RankUpdateEuclideanMetric(sz::Tuple{Int}) = RankUpdateEuclideanMetric(Float64, sz)

renew(::RankUpdateEuclideanMetric, (M⁻¹, B, D)) = RankUpdateEuclideanMetric(M⁻¹, B, D)

Base.size(metric::RankUpdateEuclideanMetric, dim...) = size(metric.M⁻¹.diag, dim...)

function Base.show(io::IO, ::MIME"text/plain", metric::RankUpdateEuclideanMetric)
    return print(io, "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))")
end

# `rand` functions for `metric` types.

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::UnitEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    return r
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DiagEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    r ./= metric.sqrtM⁻¹
    return r
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    r = _randn(rng, T, size(metric)...)
    ldiv!(metric.cholM⁻¹, r)
    return r
end

function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::RankUpdateEuclideanMetric{T},
    kinetic::GaussianKinetic,
    ::AbstractVecOrMat,
) where {T}
    M⁻¹ = metric.M⁻¹
    r = _randn(rng, T, size(M⁻¹.diag)...)
    F = metric.factorization
    k = min(size(F.U, 1), size(F.V, 1))
    @views ldiv!(F.V, r isa AbstractVector ? r[1:k] : r[1:k, :])
    lmul!(F.Q, r)
    ldiv!(F.U, r)
    return r
end
