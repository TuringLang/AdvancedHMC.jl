##########################
### Variance estimator ###
##########################

abstract type VarEstimator{T} end

# NOTE: this naive variance estimator is used only in testing
mutable struct NaiveVar{T} <: VarEstimator{T}
    n :: Int
    S :: Vector{Vector{T}}
end

NaiveVar() = NaiveVar(0, Vector{Vector{Float64}}())
NaiveVar(::Int) = NaiveVar()

function add_sample!(nc::NaiveVar, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveVar{T}) where {T<:AbstractFloat}
    nc.n = 0
    nc.S = Vector{Vector{T}}()
end

function get_var(nc::NaiveVar{T})::Vector{T} where {T<:AbstractFloat}
    @assert nc.n >= 2 "Cannot get variance with only one sample"
    return Statistics.var(nc.S)
end

# Ref： https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_var_estimator.hpp
mutable struct WelfordVar{T<:AbstractFloat,AT<:AbstractVector{T}} <: VarEstimator{T}
    n :: Int
    μ :: AT
    M :: AT
end

WelfordVar(d::Int) = WelfordVar(0, zeros(d), zeros(d))

function reset!(wv::WelfordVar{T,AT}) where {T<:AbstractFloat,AT<:AbstractVector{T}}
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
function get_var(wv::WelfordVar{T,AT})::AT where {T<:AbstractFloat,AT<:AbstractVector{T}}
    n, M = wv.n, wv.M
    @assert n >= 2 "Cannot get covariance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M .+ 1e-3 * (5 / (n + 5))
end

abstract type CovarEstimator{T} end

# NOTE: this naive covariance estimator is used only in testing
mutable struct NaiveCovar{T} <: CovarEstimator{T}
    n :: Int
    S :: Vector{Vector{T}}
end

NaiveCovar() = NaiveCovar(0, Vector{Vector{Float64}}())
NaiveCovar(::Int) = NaiveCovar()

function add_sample!(nc::NaiveCovar, s::AbstractVector)
    nc.n += 1
    push!(nc.S, s)
end

function reset!(nc::NaiveCovar{T}) where {T<:AbstractFloat}
    nc.n = 0
    nc.S = Vector{Vector{T}}()
end

function get_covar(nc::NaiveCovar{T})::Matrix{T} where {T<:AbstractFloat}
    @assert nc.n >= 2 "Cannot get covariance with only one sample"
    return Statistics.cov(nc.S)
end

# Ref: https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/welford_covar_estimator.hpp
mutable struct WelfordCovar{T<:AbstractFloat} <: CovarEstimator{T}
    n :: Int
    μ :: Vector{T}
    M :: Matrix{T}
end

WelfordCovar(d::Int) = WelfordCovar(0, zeros(d), zeros(d,d))

function reset!(wc::WelfordCovar{T}) where {T<:AbstractFloat}
    wc.n = 0
    wc.μ .= zero(T)
    wc.M .= zero(T)
end

function add_sample!(wc::WelfordCovar, s::AbstractVector)
    wc.n += 1
    δ = s .- wc.μ
    wc.μ .+= δ ./ wc.n
    wc.M .+= (s .- wc.μ) * δ'
end
# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/covar_adaptation.hpp
function get_covar(wc::WelfordCovar{T})::Matrix{T} where {T<:AbstractFloat}
    n, M = wc.n, wc.M
    @assert n >= 2 "Cannot get variance with only one sample"
    return (n / ((n + 5) * (n - 1))) .* M + 1e-3 * (5 / (n + 5)) * LinearAlgebra.I
end

################
### Adapters ###
################
# TODO: integrate metric into preconditioner
abstract type AbstractPreConditioner <: AbstractAdapter end
struct UnitPreConditioner <: AbstractPreConditioner end

string(::UnitPreConditioner) = "I"
adapt!(::UnitPreConditioner, ::AbstractVector{<:Real}, ::AbstractFloat, is_update::Bool=true) = nothing
reset!(::UnitPreConditioner) = nothing
getM⁻¹(dpc::UnitPreConditioner) = nothing

mutable struct DiagPreConditioner{T<:Real,AT<:AbstractVector{T}} <: AbstractPreConditioner
    n_min   :: Int
    ve  :: VarEstimator{T}
    var :: AT
end

function DiagPreConditioner(d::Int, n_min::Int=10)
    ve = WelfordVar(0, zeros(d), zeros(d))
    return DiagPreConditioner(n_min, ve, Vector(ones(d)))
end

function string(dpc::DiagPreConditioner)
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

mutable struct DensePreConditioner{T<:AbstractFloat} <: AbstractPreConditioner
    n_min :: Int
    ce    :: CovarEstimator{T}
    covar :: Matrix{T}
end

function DensePreConditioner(d::Integer, n_min::Int=10)
    ce = WelfordCovar(d)
    # TODO: take use of the line below when we have an interface to set which pre-conditioner to use
    # ce = NaiveCovar()
    return DensePreConditioner(n_min, ce, LinearAlgebra.diagm(0 => ones(d)))
end

function string(dpc::DensePreConditioner)
    return string(LinearAlgebra.diag(dpc.covar))
end

function adapt!(dpc::DensePreConditioner, θ::AbstractVector{<:AbstractFloat}, α::AbstractFloat, is_update::Bool=true)
    add_sample!(dpc.ce, θ)
    if dpc.ce.n >= dpc.n_min && is_update
        dpc.covar .= get_covar(dpc.ce)
    end
end

reset!(dpc::DensePreConditioner) = reset!(dpc.ce)

function getM⁻¹(dpc::DensePreConditioner)
    return dpc.covar
end
