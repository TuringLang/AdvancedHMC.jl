####
#### Robust online (co-)variance estimators.
####

abstract type MassMatrixAdaptor <: AbstractAdaptor end

initialize!(adaptor::T, n_adapts::Int) where {T<:MassMatrixAdaptor} = nothing
finalize!(adaptor::T) where {T<:MassMatrixAdaptor} = nothing

## Unit mass matrix adaptor

struct UnitMassMatrix{T<:AbstractFloat} <: MassMatrixAdaptor end

Base.show(io::IO, ::UnitMassMatrix) = print(io, "UnitMassMatrix")

UnitMassMatrix(::Type{T}=Float64) where {T} = UnitMassMatrix{T}()

Base.string(::UnitMassMatrix) = "I"

Base.resize!(pc::UnitMassMatrix, θ::AbstractVecOrMat) = nothing

reset!(::UnitMassMatrix) = nothing

getM⁻¹(::UnitMassMatrix{T}) where {T} = UniformScaling{T}(one(T))

adapt!(
    ::UnitMassMatrix,
    ::AbstractVecOrMat{<:AbstractFloat},
    ::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
) = nothing

## Diagonal mass matrix adaptor

abstract type VarEstimator{T} <: MassMatrixAdaptor end

getM⁻¹(ve::VarEstimator) = ve.var

Base.string(ve::VarEstimator) = string(getM⁻¹(ve))

function adapt!(
    ve::VarEstimator,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
)
    resize!(ve, θ)
    push!(ve, θ)
    if ve.n >= ve.n_min && is_update
        ve.var .= getest(ve)
    end
end

# NOTE: this naive variance estimator is used only in testing
mutable struct NaiveVar{E<:AbstractVecOrMat{<:AbstractFloat},T<:AbstractVector{E}} <: VarEstimator{T}
    n :: Int
    S :: T
end

NaiveVar(::Type{T}, sz::Tuple{Int}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Vector{T}}())
NaiveVar(::Type{T}, sz::Tuple{Int,Int}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Matrix{T}}())

NaiveVar(sz::Tuple{Vararg{Int}}; kwargs...) = NaiveVar(Float64, sz; kwargs...)

function Base.push!(nv::NaiveVar, s::AbstractVecOrMat)
    nv.n += 1
    push!(nv.S, s)
end

function reset!(nv::NaiveVar{T}) where {T}
    nv.n = 0
    nv.S = T()
end

function getest(nv::NaiveVar)
    @assert nv.n >= 2 "Cannot estimate variance with only one sample"
    return Statistics.var(nv.S)
end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{T<:AbstractVecOrMat{<:AbstractFloat}} <: VarEstimator{T}
    n     :: Int
    n_min :: Int
    μ     :: T
    M     :: T
    δ     :: T    # cache for diff
    var   :: T    # cache for variance
end

Base.show(io::IO, ::WelfordVar) = print(io, "WelfordVar")

function WelfordVar(
    ::Type{T}, sz::Union{Tuple{Int},Tuple{Int,Int}}; 
    n_min::Int=10, var=ones(T, sz)
) where {T<:AbstractFloat}
    return WelfordVar(0, n_min, zeros(T, sz), zeros(T, sz), zeros(T, sz), var)
end

WelfordVar(sz::Tuple{Vararg{Int}}; kwargs...) = WelfordVar(Float64, sz; kwargs...)

function Base.resize!(wv::WelfordVar, θ::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    if size(θ) != size(wv.var)
        @assert wv.n == 0 "Cannot resize a var estimator when it contains samples."
        wv.μ = zeros(T, size(θ))
        wv.M = zeros(T, size(θ))
        wv.δ = zeros(T, size(θ))
        wv.var = ones(T, size(θ))
    end
end

function reset!(wv::WelfordVar{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat}
    wv.n = 0
    wv.μ .= zero(T)
    wv.M .= zero(T)
end

function Base.push!(wv::WelfordVar, s::AbstractVecOrMat)
    wv.n += 1
    @unpack δ, μ, M, n = wv
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + δ .* δ * (n - 1) / n   # eqv. to `M + (s - μ) .* δ`
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function getest(wv::WelfordVar{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat}
    @unpack n, M, var = wv
    @assert n >= 2 "Cannot estimate variance with only one sample"
    return T(n) / ((n + 5) * (n - 1)) .* M .+ T(1e-3) * (5 / (n + 5))
end

## Dense mass matrix adaptor

abstract type CovEstimator{T} <: MassMatrixAdaptor end

getM⁻¹(ce::CovEstimator) = ce.cov

Base.string(ce::CovEstimator) = string(LinearAlgebra.diag(getM⁻¹(ce)))

function adapt!(
    ce::CovEstimator,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
)
    resize!(ce, θ)
    push!(ce, θ)
    if ce.n >= ce.n_min && is_update
        ce.cov .= getest(ce)
    end
end

# NOTE: This naive covariance estimator is used only in testing.
mutable struct NaiveCov{T<:AbstractVector{<:AbstractVector{<:AbstractFloat}}} <: CovEstimator{T}
    n :: Int
    S :: T
end

NaiveCov(::Type{T}, sz::Tuple{Int}) where {T<:AbstractFloat} = NaiveCov(0, Vector{Vector{T}}())

NaiveCov(sz::Tuple{Vararg{Int}}; kwargs...) = NaiveCov(Float64, sz; kwargs...)

function Base.push!(nc::NaiveCov, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveCov{T}) where {T}
    nc.n = 0
    nc.S = T()
end

function getest(nc::NaiveCov)
    @assert nc.n >= 2 "Cannot get covariance with only one sample"
    return Statistics.cov(nc.S)
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCov{T<:AbstractFloat} <: CovEstimator{T}
    n     :: Int
    n_min :: Int
    μ     :: Vector{T}
    M     :: Matrix{T}
    δ     :: Vector{T} # cache for diff
    cov   :: Matrix{T}
end

Base.show(io::IO, ::WelfordCov) = print(io, "WelfordCov")

function WelfordCov(
    ::Type{T}, sz::Tuple{Int}; n_min::Int=10, cov=LinearAlgebra.diagm(0 => ones(T, first(sz)))
) where {T<:AbstractFloat}
    d = first(sz)
    return WelfordCov(0, n_min, zeros(T, d), zeros(T, d, d), zeros(T, d), cov)
end

WelfordCov(sz::Tuple{Int}; kwargs...) = WelfordCov(Float64, sz; kwargs...)

function Base.resize!(wc::WelfordCov, θ::AbstractVector{T}) where {T<:AbstractFloat}
    if length(θ) != size(wc.cov, 1)
        @assert wc.n == 0 "Cannot resize a var estimator when it contains samples."
        wc.μ = zeros(T, length(θ))
        wc.δ = zeros(T, length(θ))
        wc.M = zeros(T, length(θ), length(θ))
        wc.cov = LinearAlgebra.diagm(0 => ones(T, length(θ)))
    end
end

function reset!(wc::WelfordCov{T}) where {T<:AbstractFloat}
    wc.n = 0
    wc.μ .= zero(T)
    wc.M .= zero(T)
end

function Base.push!(wc::WelfordCov, s::AbstractVector)
    wc.n += 1
    @unpack δ, μ, n, M = wc
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + δ * δ' * (n - 1) / n   # eqv. to `M + (s - μ) * δ'
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function getest(wc::WelfordCov{T}) where {T<:AbstractFloat}
    @unpack n, M, cov = wc
    @assert n >= 2 "Cannot get covariance with only one sample"
    return T(n) / ((n + 5) * (n - 1)) .* M + T(1e-3) * (5 / (n + 5)) * LinearAlgebra.I
end

####
#### Metric
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
Base.size(e::UnitEuclideanMetric, dim::Int) = e.size[dim]
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

Base.size(e::DiagEuclideanMetric, dim...) = size(e.M⁻¹, dim...)
Base.show(io::IO, dem::DiagEuclideanMetric) = print(io, "DiagEuclideanMetric($(_string_M⁻¹(dem.M⁻¹)))")

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
function DenseEuclideanMetric(M⁻¹::Union{AbstractMatrix{T},AbstractArray{T,3}}) where {T<:AbstractFloat}
    _temp = Vector{T}(undef, Base.front(size(M⁻¹)))
    return DenseEuclideanMetric(M⁻¹, cholesky(Symmetric(M⁻¹)).U, _temp)
end
DenseEuclideanMetric(::Type{T}, D::Int) where {T} = DenseEuclideanMetric(Matrix{T}(I, D, D))
DenseEuclideanMetric(D::Int) = DenseEuclideanMetric(Float64, D)
DenseEuclideanMetric(::Type{T}, sz::Tuple{Int}) where {T} = DenseEuclideanMetric(Matrix{T}(I, first(sz), first(sz)))
DenseEuclideanMetric(sz::Tuple{Int}) = DenseEuclideanMetric(Float64, D)

renew(ue::DenseEuclideanMetric, M⁻¹) = DenseEuclideanMetric(M⁻¹)

Base.size(e::DenseEuclideanMetric, dim...) = size(e._temp, dim...)
Base.show(io::IO, dem::DenseEuclideanMetric) = print(io, "DenseEuclideanMetric(diag=$(_string_M⁻¹(dem.M⁻¹)))")

# `rand` functions for `metric` types.

function _rand(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    metric::UnitEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    return r
end

function _rand(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    metric::DiagEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    r ./= metric.sqrtM⁻¹
    return r
end

function _rand(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    metric::DenseEuclideanMetric{T}
) where {T}
    r = randn(rng, T, size(metric)...)
    ldiv!(metric.cholM⁻¹, r)
    return r
end

Base.rand(rng::AbstractRNG, metric::AbstractMetric) = _rand(rng, metric)    # this disambiguity is required by Random.rand
Base.rand(rng::AbstractVector{<:AbstractRNG}, metric::AbstractMetric) = _rand(rng, metric)
Base.rand(metric::AbstractMetric) = rand(GLOBAL_RNG, metric)

####
#### WelfordEstimator constructors
####

WelfordEstimator(m::UnitEuclideanMetric{T}) where {T} = UnitMassMatrix(T)
WelfordEstimator(m::DiagEuclideanMetric{T}) where {T} = WelfordVar(T, size(m); var=copy(m.M⁻¹))
WelfordEstimator(m::DenseEuclideanMetric{T}) where {T} = WelfordCov(T, size(m); cov=copy(m.M⁻¹))

@deprecate Preconditioner(metric) WelfordEstimator(metric)

WelfordEstimator(m::Type{TM}, sz::Tuple{Vararg{Int}}=(2,)) where {TM<:AbstractMetric} = WelfordEstimator(Float64, m, sz)
WelfordEstimator(::Type{T}, ::Type{TM}, sz::Tuple{Vararg{Int}}=(2,)) where {T<:AbstractFloat, TM<:AbstractMetric} = WelfordEstimator(TM(T, sz))
