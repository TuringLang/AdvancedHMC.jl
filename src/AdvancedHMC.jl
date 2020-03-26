module AdvancedHMC

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
using StatsFuns: logaddexp, logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter: ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck

import StatsBase: sample

### Random

# Support of passing a vector of RNGs
Base.rand(rng::AbstractVector{<:AbstractRNG}) = rand.(rng)
Base.randn(rng::AbstractVector{<:AbstractRNG}) = randn.(rng)
function Base.rand(rng::AbstractVector{<:AbstractRNG}, T, n_chains::Int)
    @argcheck length(rng) == n_chains
    return rand.(rng, T)
end
function Base.randn(rng::AbstractVector{<:AbstractRNG}, T, dim::Int, n_chains::Int)
    @argcheck length(rng) == n_chains
    return cat(randn.(rng, T, dim)...; dims=2)
end

randcat_logp(rng::AbstractRNG, unnorm_ℓp::AbstractVector) =
    randcat(rng, exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp)))
function randcat(rng::AbstractRNG, p::AbstractVector{T}) where {T}
    u = rand(rng, T)
    c = zero(eltype(p))
    i = 0
    while c < u
        c += p[i+=1]
    end
    return max(i, 1)
end

randcat_logp(rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, unnorm_ℓP::AbstractMatrix) =
    randcat(rng, exp.(unnorm_ℓP .- logsumexp(unnorm_ℓP; dims=2)))
function randcat(rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}}, P::AbstractMatrix{T}) where {T}
    u = rand(rng, T, size(P, 1))
    C = cumsum(P; dims=2)
    is = convert.(Int, vec(sum(C .< u; dims=2)))
    return max.(is, 1)
end

# Notations
# ℓπ: log density of the target distribution
# ∂ℓπ∂θ: gradient of the log density of the target distribution
# θ: position variables / model parameters
# r: momentum variables

include("adaptation/Adaptation.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
export NesterovDualAveraging, WelfordEstimator, Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor
using .Adaptation

using .Adaptation: AbstractScalarOrVec
import .Adaptation: adapt!, NesterovDualAveraging

include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog
include("trajectory.jl")
export StaticTrajectory, HMCDA, NUTS, EndPointTS, SliceTS, MultinomialTS, ClassicNoUTurn, GeneralisedNoUTurn, find_good_eps

include("diagnosis.jl")
include("sampler.jl")
export sample

### Init

using Requires

function __init__()
    include(joinpath(@__DIR__, "contrib/diffeq.jl"))
    include(joinpath(@__DIR__, "contrib/ad.jl"))
end

end # module
