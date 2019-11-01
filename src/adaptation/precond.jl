####
#### Robust online (co-)variance estimators.
####

## Variance estimator

abstract type VarEstimator{T} end

# NOTE: this naive variance estimator is used only in testing
mutable struct NaiveVar{E<:AbstractVecOrMat{<:AbstractFloat},T<:AbstractVector{E}} <: VarEstimator{T}
    n :: Int
    S :: T
end

NaiveVar(::Type{T}, ::Union{Int,Tuple{Int}}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Vector{T}}())
NaiveVar(::Type{T}, sz::Tuple{Int,Int}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Matrix{T}}())
# If `sz` are a tuple of two integers, e.g. (10, 2),
# the adaptor will estimate variance for each column (2 in this case).
NaiveVar(sz::Union{Tuple{Vararg{Int}}, Int}) = NaiveVar(Float64, sz)
NaiveVar() = NaiveVar(Float64, 0)

function add_sample!(nc::NaiveVar, s::AbstractVecOrMat)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveVar{T}) where {T}
    nc.n = 0
    nc.S = T()
end

function get_var(nc::NaiveVar)
    @assert nc.n >= 2 "Cannot get variance with only one sample"
    return Statistics.var(nc.S)
end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{T<:AbstractVecOrMat{<:AbstractFloat}} <: VarEstimator{T}
    n :: Int
    μ :: T
    M :: T
    # TODO: implement temporary `δ` as `WelfordCov`
end

WelfordVar(::Type{T}, sz::Union{Int,Tuple{Int},Tuple{Int,Int}}) where {T} = WelfordVar(0, zeros(T, sz), zeros(T, sz))
WelfordVar(sz::Union{Int,Tuple{Int},Tuple{Int,Int}}) = WelfordVar(Float64, sz)

function reset!(wv::WelfordVar{VT}) where {VT}
    T = VT |> eltype
    wv.n = 0
    wv.μ .= zero(T)
    wv.M .= zero(T)
end

function add_sample!(wv::WelfordVar, s::AbstractVecOrMat)
    wv.n += 1
    @unpack μ, M, n = wv
    for i in eachindex(s)
        δ = s[i] - μ[i]
        μ[i] += δ / n
        M[i] += δ * (s[i] - μ[i])
    end
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_var(wv::WelfordVar{VT}) where {VT}
    n, M = T(wv.n), wv.M
    @assert n >= 2 "Cannot get covariance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M .+ T(1e-3) * (5 / (n + 5))
end

## Covariance estimator

abstract type CovEstimator{T} end

# NOTE: This naive covariance estimator is used only in testing.
mutable struct NaiveCov{T<:AbstractVector{<:AbstractVector{<:AbstractFloat}}} <: CovEstimator{T}
    n :: Int
    S :: T
end

NaiveCov(::Type{T}, sz::Union{Int,Tuple{Int}}) where {T} = NaiveCov(0, Vector{Vector{T}}())
NaiveCov(sz::Union{Int,Tuple{Int}}) = NaiveCov(Float64, sz)
NaiveCov() = NaiveCov(Float64, 0)

function add_sample!(nc::NaiveCov, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function get_cov(nc::NaiveCov)
    @assert nc.n >= 2 "Cannot get covariance with only one sample"
    return Statistics.cov(nc.S)
end

function reset!(nc::NaiveCov{T}) where {T}
    nc.n = 0
    nc.S = T()
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCov{T<:AbstractFloat} <: CovEstimator{T}
    n :: Int
    μ :: Vector{T}
    M :: Matrix{T}
    δ :: Vector{T} # temporary
end

function WelfordCov(::Type{T}, d::Int) where {T}
    return WelfordCov(0, zeros(T, d), zeros(T, d, d), zeros(T, d))
end
WelfordCov(d::Int) = WelfordCov(Float64, d)
WelfordCov(type::Type{T}, sz::Tuple{Int}) where {T} = WelfordCov(type, first(sz))

function reset!(wc::WelfordCov{T}) where {T<:AbstractFloat}
    wc.n = 0
    wc.μ .= zero(T)
    wc.M .= zero(T)
end

function add_sample!(wc::WelfordCov, s::AbstractVector)
    wc.n += 1
    @unpack δ, μ, n, M = wc
    δ .= s .- μ
    μ .+= δ ./ n
    M .+= (s .- μ) .* δ'
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_cov(wc::WelfordCov{T}) where {T<:AbstractFloat}
    n, M = T(wc.n), wc.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M + T(1e-3) * (5 / (n + 5)) * LinearAlgebra.I
end

###
### Preconditioning matrix adaptors.
###

abstract type AbstractPreconditioner <: AbstractAdaptor end

finalize!(adaptor::T) where {T<:AbstractPreconditioner} = nothing

####
#### Preconditioning matrix adaption implementation.
####

# Unit
struct UnitPreconditioner{T} <: AbstractPreconditioner end
Base.show(io::IO, ::UnitPreconditioner) = print(io, "UnitPreconditioner")

UnitPreconditioner(::Type{T}=Float64) where {T} = UnitPreconditioner{T}()

string(::UnitPreconditioner) = "I"
reset!(::UnitPreconditioner) = nothing
getM⁻¹(dpc::UnitPreconditioner{T}) where {T<:AbstractFloat} = UniformScaling{T}(one(T))
adapt!(
    ::UnitPreconditioner,
    ::AbstractVecOrMat{<:AbstractFloat},
    ::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
) = nothing


mutable struct DiagPreconditioner{
    AT<:AbstractVecOrMat{<:AbstractFloat},
    TEst<:VarEstimator{AT}
} <: AbstractPreconditioner
    n_min   :: Int
    ve      :: TEst
    var     :: AT
end
Base.show(io::IO, ::DiagPreconditioner) = print(io, "DiagPreconditioner")

# Diagonal
DiagPreconditioner(sz::Tuple{Vararg{Int}}, n_min::Int=10) = DiagPreconditioner(Float64, sz, n_min)
function DiagPreconditioner(
    ::Type{T},
    sz::Tuple{Vararg{Int}};
    n_min::Int=10,
    var=ones(T, sz)
) where {T}
    ve = WelfordVar(T, sz)
    return DiagPreconditioner(n_min, ve, var)
end

string(dpc::DiagPreconditioner) = string(dpc.var)
reset!(dpc::DiagPreconditioner) = reset!(dpc.ve)
getM⁻¹(dpc::DiagPreconditioner) = dpc.var

function adapt!(
    dpc::DiagPreconditioner,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
)
    resize!(dpc, θ)
    add_sample!(dpc.ve, θ)
    if dpc.ve.n >= dpc.n_min && is_update
        # Can be made inplace
        dpc.var .= get_var(dpc.ve)
    end
end

# Dense
mutable struct DensePreconditioner{
    T<:AbstractFloat,
    TEst<:CovEstimator{T}
} <: AbstractPreconditioner
    n_min :: Int
    ce    :: TEst
    covar :: Matrix{T}
end
Base.show(io::IO, ::DensePreconditioner) = print(io, "DensePreconditioner")

DensePreconditioner(sz::Tuple{Int}, n_min::Int=10) = DensePreconditioner(Float64, sz, n_min)
function DensePreconditioner(
    ::Type{T},
    sz::Tuple{Int};
    n_min::Int=10,
    covar=LinearAlgebra.diagm(0 => ones(T, sz))
) where {T}
    ce = WelfordCov(T, sz)
    # TODO: take use of the line below when we have an interface to set which covariance estimator to use
    # ce = NaiveCov(T)
    return DensePreconditioner(n_min, ce, covar)
end

string(dpc::DensePreconditioner) = string(LinearAlgebra.diag(dpc.covar))
reset!(dpc::DensePreconditioner) = reset!(dpc.ce)
getM⁻¹(dpc::DensePreconditioner) = dpc.covar

function adapt!(
    dpc::DensePreconditioner,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
)
    resize!(dpc, θ)
    add_sample!(dpc.ce, θ)
    if dpc.ce.n >= dpc.n_min && is_update
        # Can be made inplace
        dpc.covar .= get_cov(dpc.ce)
    end
end

# Resize pre-conditioner if necessary.
Base.resize!(
    pc::UnitPreconditioner,
    θ::AbstractVecOrMat
) = nothing

function Base.resize!(
    dpc::DiagPreconditioner,
    θ::AbstractVecOrMat{T}
) where {T<:AbstractFloat}
    if size(θ) != size(dpc.var)
        @assert dpc.ve.n == 0 "Cannot resize a var estimator when it contains samples."
        dpc.ve = WelfordVar(T, size(θ))
        dpc.var = ones(T, size(θ))
    end
end

function Base.resize!(
    dpc::DensePreconditioner,
    θ::AbstractVecOrMat{T}
) where {T<:AbstractFloat}
    if length(θ) != size(dpc.covar,1)
        @assert dpc.ce.n == 0 "Cannot resize a var estimator when it contains samples."
        dpc.ce = WelfordCov(T, length(θ))
        dpc.covar = LinearAlgebra.diagm(0 => ones(T, length(θ)))
    end
end

####
#### Preconditioning mass matrix.
####

abstract type AbstractMetric end

_string_M⁻¹(mat::AbstractMatrix, n_chars::Int=32) = _string_M⁻¹(LinearAlgebra.diag(mat), n_chars)
function _string_M⁻¹(vec::AbstractVector, n_chars::Int=32)
    s_vec = string(vec)
    l = length(s_vec)
    s_dots = " ...]"
    n_diag_chars = n_chars - length(s_dots)
    return s_vec[1:min(n_diag_chars,end)] * (l > n_diag_chars ? s_dots : "")
end

struct UnitEuclideanMetric{T,A<:Union{Tuple{Int},Tuple{Int,Int}}} <: AbstractMetric
    M⁻¹::UniformScaling{T}
    size::A
end

UnitEuclideanMetric(::Type{T}, sz) where {T} = UnitEuclideanMetric(UniformScaling{T}(one(T)), sz)
UnitEuclideanMetric(sz) = UnitEuclideanMetric(Float64, sz)
UnitEuclideanMetric(::Type{T}, dim::Int) where {T} = UnitEuclideanMetric(UniformScaling{T}(one(T)), (dim,))
UnitEuclideanMetric(dim::Int) = UnitEuclideanMetric(Float64, (dim,))

renew(ue::UnitEuclideanMetric, M⁻¹) = UnitEuclideanMetric(M⁻¹, ue.size)

Base.size(e::UnitEuclideanMetric) = e.size
Base.show(io::IO, uem::UnitEuclideanMetric) = print(io, "UnitEuclideanMetric($(_string_M⁻¹(ones(uem.size))))")

struct DiagEuclideanMetric{T,A<:AbstractVecOrMat{T}} <: AbstractMetric
    # Diagnal of the inverse of the mass matrix
    M⁻¹     ::  A
    # Sqare root of the inverse of the mass matrix
    sqrtM⁻¹ ::  A
    # Pre-allocation for intermediate variables
    _temp   ::  A
end

function DiagEuclideanMetric(M⁻¹::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    return DiagEuclideanMetric(M⁻¹, sqrt.(M⁻¹), similar(M⁻¹))
end
DiagEuclideanMetric(::Type{T}, sz) where {T} = DiagEuclideanMetric(ones(T, sz...))
DiagEuclideanMetric(sz) = DiagEuclideanMetric(Float64, sz)
DiagEuclideanMetric(::Type{T}, dim::Int) where {T} = DiagEuclideanMetric(ones(T, dim))
DiagEuclideanMetric(dim::Int) = DiagEuclideanMetric(Float64, dim)

renew(ue::DiagEuclideanMetric, M⁻¹) = DiagEuclideanMetric(M⁻¹)

Base.size(e::DiagEuclideanMetric) = size(e.M⁻¹)
Base.show(io::IO, dem::DiagEuclideanMetric) = print(io, "DiagEuclideanMetric($(_string_M⁻¹(dem.M⁻¹)))")

struct DenseEuclideanMetric{
    T,
    AV<:AbstractVector{T},
    AM<:AbstractMatrix{T},
    TcholM⁻¹<:UpperTriangular{T},
} <: AbstractMetric
    # Inverse of the mass matrix
    M⁻¹::AM
    # U of the Cholesky decomposition of the mass matrix
    cholM⁻¹::TcholM⁻¹
    # Pre-allocation for intermediate variables
    _temp::AV
end

function DenseEuclideanMetric(M⁻¹::AbstractMatrix{T}) where {T<:AbstractFloat}
    _temp = Vector{T}(undef, size(M⁻¹, 1))
    return DenseEuclideanMetric(M⁻¹, cholesky(Symmetric(M⁻¹)).U, _temp)
end
DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T} = DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, D)

renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

function Base.size(e::DenseEuclideanMetric)
    sz = size(e.M⁻¹)
    if length(sz) == 2
        # If `M⁻¹` stores a D x D tensor, we would like to return only D
        sz = (sz[2],)
    elseif length(sz) == 3
        # If `M⁻¹` stores a D x D x C tensor, we would like to return only D x C
        sz = (sz[2], sz[3])
    end
    return sz
end
Base.show(io::IO, dem::DenseEuclideanMetric) = print(io, "DenseEuclideanMetric(diag=$(_string_M⁻¹(dem.M⁻¹)))")

# `rand` functions for `metric` types.
function Base.rand(
    rng::AbstractRNG,
    metric::UnitEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    return r
end

function Base.rand(
    rng::AbstractRNG,
    metric::DiagEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    r ./= metric.sqrtM⁻¹
    return r
end

function Base.rand(
    rng::AbstractRNG,
    metric::DenseEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    ldiv!(metric.cholM⁻¹, r)
    return r
end

Base.rand(metric::AbstractMetric) = rand(GLOBAL_RNG, metric)

####
#### Preconditioner constructors
####

Preconditioner(m::UnitEuclideanMetric{T}) where {T} = UnitPreconditioner(T)
Preconditioner(m::DiagEuclideanMetric{T}) where {T} = DiagPreconditioner(T, size(m); var=m.M⁻¹)
Preconditioner(m::DenseEuclideanMetric{T}) where {T} = DensePreconditioner(T, size(m); covar=m.M⁻¹)

Preconditioner(m::Type{TM}, sz::Tuple{Vararg{Int}}=(2,)) where {TM<:AbstractMetric} = Preconditioner(Float64, m, sz)
Preconditioner(::Type{T}, ::Type{TM}, sz::Tuple{Vararg{Int}}=(2,)) where {T<:AbstractFloat, TM<:AbstractMetric} = Preconditioner(TM(T, sz))
