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
reset!(::UnitPreConditioner) = nothing

mutable struct DiagPreConditioner{T<:Real,AT<:AbstractVector{T}} <: AbstractPreConditioner
    n_min   :: Int
    ve  :: VarEstimator{T}
    var :: AT
end

function DiagPreConditioner(d::Int, n_min::Int=10)
    ve = WelfordVar(0, zeros(d), zeros(d))
    return DiagPreConditioner(n_min, ve, Vector(ones(d)))
end

function Base.string(dpc::DiagPreConditioner)
    return string(dpc.var)
end

function adapt!(dpc::DiagPreConditioner, θ::AbstractVector{<:Real}, α::AbstractFloat, is_update::Bool=true)
    add_sample!(dpc.ve, θ)
    if dpc.ve.n >= dpc.n_min && is_update
        dpc.var .= get_var(dpc.ve)
    end
end

reset!(dpc::DiagPreConditioner) = reset!(dpc.ve)

function getM⁻¹(dpc::DiagPreConditioner)
    return dpc.var
end

mutable struct DensePreConditioner{TI<:Integer,TF<:Real} <: AbstractPreConditioner
    n_min :: Int
    ce    :: CovarEstimator{TI,TF}
    covar :: Matrix{TF}
end

function DensePreConditioner(d::Integer, n_min::Int=10)
    ce = WelfordCovar(0, zeros(d), zeros(d,d))
    # TODO: take use of the line below when we have an interface to set which pre-conditioner to use
    # ce = NaiveCovar(0, Vector{Vector{Float64}}())
    return DensePreConditioner(n_min, ce, LinearAlgebra.diagm(0 => ones(d)))
end

function Base.string(dpc::DensePreConditioner)
    return string(LinearAlgebra.diag(dpc.covar))
end

function adapt!(dpc::DensePreConditioner, θ::AbstractVector{<:Real}, α::AbstractFloat, is_update::Bool=true)
    add_sample!(dpc.ce, θ)
    if dpc.ce.n >= dpc.n_min && is_update
        dpc.covar .= get_covar(dpc.ce)
    end
end

reset!(dpc::DensePreConditioner) = reset!(dpc.ce)

function getM⁻¹(dpc::DensePreConditioner)
    return dpc.covar
end
