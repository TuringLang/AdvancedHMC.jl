"""
$(TYPEDEF)

Abstract type for preconditioning metrics. 
"""
abstract type AbstractMetric end

_string_M⁻¹(mat::AbstractMatrix, n_chars::Int=32) = _string_M⁻¹(diag(mat), n_chars)
function _string_M⁻¹(vec::AbstractVector, n_chars::Int=32)
    s_vec = repr(vec; context=(:compact => true))
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

function Base.show(io::IO, uem::UnitEuclideanMetric{T}) where {T}
    return print(io, "UnitEuclideanMetric(", T, ", ", uem.size, ")")
end
function Base.show(io::IO, ::MIME"text/plain", uem::UnitEuclideanMetric{T}) where {T}
    return print(
        io,
        "UnitEuclideanMetric{",
        T,
        "} with size ",
        size(uem),
        " mass matrix:\n",
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

function Base.show(io::IO, dem::DiagEuclideanMetric)
    return print(io, "DiagEuclideanMetric(", _string_M⁻¹(dem.M⁻¹), ")")
end
function Base.show(io::IO, ::MIME"text/plain", dem::DiagEuclideanMetric{T}) where {T}
    return print(
        io,
        "DiagEuclideanMetric{",
        T,
        "} with size ",
        size(dem),
        " mass matrix:\n",
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

function Base.show(io::IO, dem::DenseEuclideanMetric)
    return print(io, "DenseEuclideanMetric(", _string_M⁻¹(dem.M⁻¹), ")")
end
function Base.show(io::IO, ::MIME"text/plain", dem::DenseEuclideanMetric{T}) where {T}
    return print(
        io,
        "DenseEuclideanMetric{",
        T,
        "} with size ",
        size(dem),
        " mass matrix:\n",
        _string_M⁻¹(dem.M⁻¹),
    )
end

"""
    WoodburyFactorization(U, Q, V)

A factorization of a positive definite Woodbury matrix `W = A + B*D*Bᵀ`, with positive
definite diagonal `A`, as returned by [`woodbury_factorize`](@ref).

The factors `U`, `Q`, and `V` are defined by the field descriptions below; together they
allow sampling from `N(0, W⁻¹)` without forming `W`.

# Fields

$(TYPEDFIELDS)
"""
struct WoodburyFactorization{T,TU<:Diagonal{T},TQ<:AbstractQ{T},TV<:UpperTriangular{T}}
    "diagonal Cholesky factor of `A`, i.e. `Uᵀ*U = A`"
    U::TU
    "orthogonal factor of the thin QR decomposition `Uᵀ \\ B = Q*R`"
    Q::TQ
    "upper-triangular Cholesky factor of `I + R*D*Rᵀ`, i.e. `Vᵀ*V = I + R*D*Rᵀ`"
    V::TV
end

"""
    woodbury_factorize(A::Diagonal, B::AbstractMatrix, D::AbstractMatrix)

Return a [`WoodburyFactorization`](@ref) of the positive definite Woodbury matrix
`W = A + B*D*Bᵀ`, where `A` is a positive definite diagonal matrix.
"""
function woodbury_factorize(
    A::Diagonal{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    U = cholesky(A).U
    Q, R = qr(U' \ B)
    V = cholesky(Symmetric(muladd(R, D * R', I))).U
    return WoodburyFactorization(U, Q, V)
end

"""
    RankUpdateEuclideanMetric([T::Type=Float64,] n::Int)
    RankUpdateEuclideanMetric([T::Type=Float64,] sz::Tuple{Int})
    RankUpdateEuclideanMetric(A::Diagonal, B::AbstractMatrix, D::AbstractMatrix)

A Gaussian Euclidean metric in `n` dimensions whose inverse mass matrix `M⁻¹ = A + B*D*Bᵀ`
is a low-rank update of a positive definite diagonal matrix `A`.

The rank `k` of the update equals the number of columns of `B`. Evaluating the kinetic
energy and its gradient then costs `O(n*k)`, compared with `O(n^2)` for a dense metric, so
the metric is useful when most of the posterior covariance lies in a low-dimensional
subspace. As for the other metrics, `M⁻¹` denotes the full inverse mass matrix; here it is
reconstructed from the fields rather than stored explicitly.

`RankUpdateEuclideanMetric(n)` constructs an `n`-by-`n` metric with `M⁻¹` equal to the
identity (a rank-0 update). `RankUpdateEuclideanMetric(A, B, D)` constructs the metric from
a positive definite `Diagonal` matrix `A` and matrices `B` and `D` of matching element type,
chosen such that `M⁻¹` is positive definite. The element type `T` defaults to `Float64`.

# Fields

$(TYPEDFIELDS)

# Reference

  - Zhang, Carpenter, Gelman & Vehtari (2022). Pathfinder: Parallel quasi-Newton variational
    inference. Journal of Machine Learning Research 23(306), 1-49.
"""
struct RankUpdateEuclideanMetric{
    T,
    AA<:Diagonal{T},
    AB<:AbstractMatrix{T},
    AD<:AbstractMatrix{T},
    F<:WoodburyFactorization{T},
} <: AbstractMetric
    "positive definite diagonal matrix `A` in `M⁻¹ = A + B*D*Bᵀ`"
    A::AA
    "factor `B` of the low-rank update `B*D*Bᵀ`"
    B::AB
    "inner matrix `D` of the low-rank update `B*D*Bᵀ`"
    D::AD
    "[`WoodburyFactorization`](@ref) of `M⁻¹`, used for momentum sampling"
    factorization::F
end

function RankUpdateEuclideanMetric(::Type{T}, n::Int) where {T}
    A = Diagonal(ones(T, n))
    B = Matrix{T}(undef, n, 0)
    D = Matrix{T}(undef, 0, 0)
    # For the identity (rank-0) metric the Woodbury factors are trivial: `U = A`, `Q` is the
    # identity, and `V` is empty, so we build them directly instead of via `cholesky`.
    factorization = WoodburyFactorization(A, qr(B).Q, UpperTriangular(D))
    return RankUpdateEuclideanMetric(A, B, D, factorization)
end
RankUpdateEuclideanMetric(n::Int) = RankUpdateEuclideanMetric(Float64, n)

function RankUpdateEuclideanMetric(
    A::Diagonal{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    return RankUpdateEuclideanMetric(A, B, D, woodbury_factorize(A, B, D))
end

function RankUpdateEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T}
    return RankUpdateEuclideanMetric(T, first(sz))
end
RankUpdateEuclideanMetric(sz::Tuple{Int}) = RankUpdateEuclideanMetric(Float64, sz)

function renew(::RankUpdateEuclideanMetric, (A, B, D))
    return RankUpdateEuclideanMetric(A, B, D)
end

Base.eltype(::RankUpdateEuclideanMetric{T}) where {T} = T
Base.size(metric::RankUpdateEuclideanMetric, dim...) = size(metric.A.diag, dim...)

# Diagonal of the full inverse mass matrix `M⁻¹ = A + B*D*Bᵀ`, i.e. `diag(A) + [bᵢᵀ*D*bᵢ]`.
function _diag_inv_metric(metric::RankUpdateEuclideanMetric)
    return broadcast(
        (aii, b) -> aii + dot(b, metric.D, b), metric.A.diag, eachrow(metric.B)
    )
end

function Base.show(io::IO, metric::RankUpdateEuclideanMetric)
    return print(
        io, "RankUpdateEuclideanMetric(", _string_M⁻¹(_diag_inv_metric(metric)), ")"
    )
end
function Base.show(
    io::IO, ::MIME"text/plain", metric::RankUpdateEuclideanMetric{T}
) where {T}
    return print(
        io,
        "RankUpdateEuclideanMetric{",
        T,
        "} with size ",
        size(metric),
        " mass matrix:\n",
        _string_M⁻¹(_diag_inv_metric(metric)),
    )
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
    (; U, Q, V) = metric.factorization
    r = _randn(rng, T, size(metric)...)
    k = min(size(U, 1), size(V, 1))
    if r isa AbstractVector
        ldiv!(V, @view(r[begin:(begin + (k - 1))]))
    else
        ldiv!(V, @view(r[begin:(begin + (k - 1)), :]))
    end
    lmul!(Q, r)
    ldiv!(U, r)
    return r
end
