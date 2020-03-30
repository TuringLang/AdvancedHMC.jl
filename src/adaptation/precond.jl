####
#### Robust online (co-)variance estimators.
####

abstract type MassMatrixAdaptor <: AbstractAdaptor end

initialize!(::MassMatrixAdaptor, ::Int) = nothing
finalize!(::MassMatrixAdaptor) = nothing

## Unit mass matrix adaptor

struct UnitMassMatrix{T<:AbstractFloat} <: MassMatrixAdaptor end

Base.show(io::IO, ::UnitMassMatrix) = print(io, "UnitMassMatrix")

UnitMassMatrix() = UnitMassMatrix{Float64}()

Base.string(::UnitMassMatrix) = "I"

Base.resize!(pc::UnitMassMatrix, θ::AbstractVecOrMat) = nothing

reset!(::UnitMassMatrix) = nothing

getM⁻¹(::UnitMassMatrix{T}) where {T} = LinearAlgebra.UniformScaling{T}(one(T))

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
struct NaiveVar{S<:AbstractFloat,T<:AbstractVector{<:AbstractVecOrMat{S}}} <: VarEstimator{T}
    n :: Int
    S :: T
end

NaiveVar(::Type{T}, sz::Tuple{Int}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Vector{T}}())
NaiveVar(::Type{T}, sz::Tuple{Int,Int}) where {T<:AbstractFloat} = NaiveVar(0, Vector{Matrix{T}}())

NaiveVar(sz::Tuple{Vararg{Int}}; kwargs...) = NaiveVar(Float64, sz; kwargs...)

function Base.push!(nv::NaiveVar, s::AbstractVecOrMat)
    push!(nv.S, s)
end

function reset!(nv::NaiveVar)
    resize!(nv.S, 0)
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

function Base.push!(wv::WelfordVar, s::AbstractVecOrMat{T}) where {T}
    wv.n += 1
    @unpack δ, μ, M, n = wv
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + δ .* δ * (T(n - 1) / n)    # eqv. to `M + (s - μ) .* δ`
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function getest(wv::WelfordVar{<:AbstractVecOrMat{T}}) where {T<:AbstractFloat}
    @unpack n, M, var = wv
    @assert n >= 2 "Cannot estimate variance with only one sample"
    return T(n) / ((n + 5) * (n - 1)) * M .+ T(1e-3) * (5 / (n + 5))
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

function Base.push!(wc::WelfordCov, s::AbstractVector{T}) where {T}
    wc.n += 1
    @unpack δ, μ, n, M = wc
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + (s - μ) * δ'
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function getest(wc::WelfordCov{T}) where {T<:AbstractFloat}
    @unpack n, M, cov = wc
    @assert n >= 2 "Cannot get covariance with only one sample"
    return T(n) / ((n + 5) * (n - 1)) * M + T(1e-3) * (5 / (n + 5)) * LinearAlgebra.I
end
