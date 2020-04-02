####
#### Robust online (co-)variance estimators.
####

abstract type MassMatrixAdaptor <: AbstractAdaptor end

initialize!(::MassMatrixAdaptor, ::Int) = nothing
finalize!(::MassMatrixAdaptor) = nothing

function adapt!(
    adaptor::MassMatrixAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    is_update::Bool=true
)
    resize!(adaptor, θ)
    push!(adaptor, θ)
    is_update && update!(adaptor)
end

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

abstract type DiagMatrixEstimator{T} <: MassMatrixAdaptor end

getM⁻¹(ve::DiagMatrixEstimator) = ve.var

Base.string(ve::DiagMatrixEstimator) = string(getM⁻¹(ve))

function update!(ve::DiagMatrixEstimator)
    ve.n >= ve.n_min && (ve.var .= get_estimation(ve))
end

# NOTE: this naive variance estimator is used only in testing
struct NaiveVar{T<:AbstractFloat, E<:AbstractVector{<:AbstractVecOrMat{T}}} <: DiagMatrixEstimator{T}
    S :: E
    NaiveVar(S::E) where {E} = new{eltype(eltype(E)), E}(S)
end

NaiveVar{T}(sz::Tuple{Int}) where {T<:AbstractFloat} = NaiveVar(Vector{Vector{T}}())
NaiveVar{T}(sz::Tuple{Int,Int}) where {T<:AbstractFloat} = NaiveVar(Vector{Matrix{T}}())

NaiveVar(sz::Union{Tuple{Int}, Tuple{Int,Int}}) = NaiveVar{Float64}(sz)

Base.push!(nv::NaiveVar, s::AbstractVecOrMat) = push!(nv.S, s)

reset!(nv::NaiveVar) = resize!(nv.S, 0)

function get_estimation(nv::NaiveVar)
    @assert length(nv.S) >= 2 "Cannot estimate variance with only one sample"
    return Statistics.var(nv.S)
end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{T<:AbstractFloat, E<:AbstractVecOrMat{T}} <: DiagMatrixEstimator{T}
    n     :: Int
    n_min :: Int
    μ     :: E
    M     :: E
    δ     :: E    # cache for diff
    var   :: E    # cache for variance
    function WelfordVar(n::Int, n_min::Int, μ::E, M::E, δ::E, var::E) where {E}
        return new{eltype(E), E}(n, n_min, μ, M, δ, var)
    end
end

Base.show(io::IO, ::WelfordVar) = print(io, "WelfordVar")

function WelfordVar{T}(
    sz::Union{Tuple{Int}, Tuple{Int,Int}}; 
    n_min::Int=10, var=ones(T, sz)
) where {T<:AbstractFloat}
    return WelfordVar(0, n_min, zeros(T, sz), zeros(T, sz), zeros(T, sz), var)
end

WelfordVar(sz::Union{Tuple{Int}, Tuple{Int,Int}}; kwargs...) = WelfordVar{Float64}(sz; kwargs...)

function Base.resize!(wv::WelfordVar, θ::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    if size(θ) != size(wv.var)
        @assert wv.n == 0 "Cannot resize a var estimator when it contains samples."
        wv.μ = zeros(T, size(θ))
        wv.M = zeros(T, size(θ))
        wv.δ = zeros(T, size(θ))
        wv.var = ones(T, size(θ))
    end
end

function reset!(wv::WelfordVar{T}) where {T<:AbstractFloat}
    wv.n = 0
    wv.μ .= zero(T)
    wv.M .= zero(T)
end

function Base.push!(wv::WelfordVar, s::AbstractVecOrMat{T}) where {T}
    wv.n += 1
    @unpack δ, μ, M, n = wv
    n = T(n)
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + δ .* δ * ((n - 1) / n)    # eqv. to `M + (s - μ) .* δ`
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_estimation(wv::WelfordVar{T}) where {T<:AbstractFloat}
    @unpack n, M, var = wv
    @assert n >= 2 "Cannot estimate variance with only one sample"
    n, ϵ = T(n), T(1e-3)
    return n / ((n + 5) * (n - 1)) * M .+ ϵ * (5 / (n + 5))
end

## Dense mass matrix adaptor

abstract type DenseMatrixEstimator{T} <: MassMatrixAdaptor end

getM⁻¹(ce::DenseMatrixEstimator) = ce.cov

Base.string(ce::DenseMatrixEstimator) = string(LinearAlgebra.diag(getM⁻¹(ce)))

function update!(ce::DenseMatrixEstimator)
    ce.n >= ce.n_min && (ce.cov .= get_estimation(ce))
end

# NOTE: This naive covariance estimator is used only in testing.
struct NaiveCov{F<:AbstractFloat, T<:AbstractVector{<:AbstractVector{F}}} <: DenseMatrixEstimator{T}
    S :: T
    NaiveCov(S::E) where {E} = new{eltype(eltype(E)), E}(S)
end

NaiveCov{T}(sz::Tuple{Int}) where {T<:AbstractFloat} = NaiveCov(Vector{Vector{T}}())

NaiveCov(sz::Union{Tuple{Int}, Tuple{Int,Int}}; kwargs...) = NaiveCov{Float64}(sz; kwargs...)

Base.push!(nc::NaiveCov, s::AbstractVector) = push!(nc.S, s)

reset!(nc::NaiveCov{T}) where {T} = resize!(nc.S, 0)

function get_estimation(nc::NaiveCov)
    @assert length(nc.S) >= 2 "Cannot get covariance with only one sample"
    return Statistics.cov(nc.S)
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCov{F<:AbstractFloat} <: DenseMatrixEstimator{F}
    n     :: Int
    n_min :: Int
    μ     :: Vector{F}
    M     :: Matrix{F}
    δ     :: Vector{F}  # cache for diff
    cov   :: Matrix{F}
end

Base.show(io::IO, ::WelfordCov) = print(io, "WelfordCov")

function WelfordCov{T}(
    sz::Tuple{Int}; n_min::Int=10, cov=LinearAlgebra.diagm(0 => ones(T, first(sz)))
) where {T<:AbstractFloat}
    d = first(sz)
    return WelfordCov(0, n_min, zeros(T, d), zeros(T, d, d), zeros(T, d), cov)
end

WelfordCov(sz::Tuple{Int}; kwargs...) = WelfordCov{Float64}(sz; kwargs...)

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
    n = T(n)
    δ .= s - μ
    μ .= μ + δ / n
    M .= M + (s - μ) * δ'
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_estimation(wc::WelfordCov{T}) where {T<:AbstractFloat}
    @unpack n, M, cov = wc
    @assert n >= 2 "Cannot get covariance with only one sample"
    n, ϵ = T(n), T(1e-3)
    return n / ((n + 5) * (n - 1)) * M + ϵ * (5 / (n + 5)) * LinearAlgebra.I
end
