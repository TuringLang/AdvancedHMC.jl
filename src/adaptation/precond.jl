# TODO: resolve type stability of this file

##########################
### Variance estimator ###
##########################
abstract type VarEstimator{T} end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{T<:Real,AT<:AbstractVector{T}} <: VarEstimator{T}
    n :: Int
    μ :: AT
    M :: AT
end

function reset!(wv::WelfordVar{T,AT}) where {T<:Real,AT<:AbstractVector{T}}
    wv.n = 0
    wv.μ .= zero(T)
    wv.M .= zero(T)
end

function add_sample!(wv::WelfordVar, s::AbstractVector)
    wv.n += 1
    δ = s .- wv.μ
    wv.μ .+= δ ./ wv.n
    wv.M .+= δ .* (s .- wv.μ)
end

# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/var_adaptation.hpp
function get_var(wv::VarEstimator{T})::Vector{T} where {T<:Real,AT<:AbstractVector{T}}
    n, M = wv.n, wv.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M .+ 1e-3 * (5 / (n + 5))
end

abstract type CovarEstimator{TI<:Integer,TF<:Real} end

# NOTE: this naive covariance estimator is used only in testing
mutable struct NaiveCovar{TI,TF} <: CovarEstimator{TI,TF}
    n :: TI
    S :: Vector{Vector{TF}}
end

function add_sample!(nc::NaiveCovar, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveCovar{TI,TF}) where {TI<:Integer,TF<:Real}
    nc.n = zero(TI)
    nc.S = Vector{Vector{TF}}()
end

function get_covar(nc::NaiveCovar{TI,TF})::Matrix{TF} where {TI<:Integer,TF<:Real}
    @assert nc.n >= 2 "Cannot get variance with only one sample"
    return Statistics.cov(nc.S)
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCovar{TI<:Integer,TF<:Real} <: CovarEstimator{TI,TF}
    n :: TI
    μ :: Vector{TF}
    M :: Matrix{TF}
end

function reset!(wc::WelfordCovar{TI,TF}) where {TI<:Integer,TF<:Real}
    wc.n = zero(TI)
    wc.μ .= zero(TF)
    wc.M .= zero(TF)
end

function add_sample!(wc::WelfordCovar, s::AbstractVector)
    wc.n += 1
    δ = s .- wc.μ
    wc.μ .+= δ ./ wc.n
    wc.M .+= (s .- wc.μ) * δ'
end
# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_covar(wc::WelfordCovar{TI,TF})::Matrix{TF} where {TI<:Integer,TF<:Real}
    n, M = wc.n, wc.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M + 1e-3 * (5 / (n + 5)) * LinearAlgebra.I
end

################
### Adapters ###
################
abstract type AbstractPreConditioner <: AbstractAdapter end
struct UnitPreConditioner <: AbstractPreConditioner end

function Base.string(::UnitPreConditioner)
    return string([1.0])
end

adapt!(::UnitPreConditioner, ::AbstractVector{<:Real}, ::AbstractFloat) = nothing

mutable struct DiagPreConditioner{T<:Real,AT<:AbstractVector{T}} <: AbstractPreConditioner
    ve  :: VarEstimator{T}
    var :: AT
    is_updated :: Bool
end

function DiagPreConditioner(d::Int)
    ve = WelfordVar(0, zeros(d), zeros(d))
    return DiagPreConditioner(ve, Vector(ones(d)), false)
end

function Base.string(dpc::DiagPreConditioner)
    return string(dpc.var)
end

function adapt!(dpc::DiagPreConditioner, θ::AbstractVector{<:Real}, α::AbstractFloat)
    add_sample!(dpc.ve, θ)
    dpc.is_updated = false
end

function getM⁻¹(dpc::DiagPreConditioner)
    if !dpc.is_updated
        dpc.var .= get_var(dpc.ve)
        dpc.is_updated = true
    end
    return dpc.var
end

mutable struct DensePreConditioner{TI<:Integer,TF<:Real} <: AbstractPreConditioner
    ce    :: CovarEstimator{TI,TF}
    covar :: Matrix{TF}
    is_updated :: Bool
end

function DensePreConditioner(d::Integer)
    ce = WelfordCovar(0, zeros(d), zeros(d,d))
    # TODO: take use of the line below when we have an interface to set which pre-conditioner to use
    # ce = NaiveCovar(0, Vector{Vector{Float64}}())
    return DensePreConditioner(ce, LinearAlgebra.diagm(0 => ones(d)), false)
end

function Base.string(dpc::DensePreConditioner)
    return string(LinearAlgebra.diag(dpc.covar))
end

function adapt!(dpc::DensePreConditioner, θ::AbstractVector{<:Real}, α::AbstractFloat, )
    add_sample!(dpc.ce, θ)
    dpc.is_updated = false
end

function getM⁻¹(dpc::DensePreConditioner)
    if !dpc.is_updated
        dpc.covar .= get_covar(dpc.ce)
        dpc.is_updated = true
    end
    return dpc.covar
end
