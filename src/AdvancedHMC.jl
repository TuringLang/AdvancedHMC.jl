module AdvancedHMC

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
using StatsFuns: logaddexp
using LazyArrays: BroadcastArray
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck

import StatsBase: sample

# Support of passing a vector of RNGs
Base.rand(rng::AbstractVector{<:AbstractRNG}) = rand.(rng)
Base.randn(rng::AbstractVector{<:AbstractRNG}) = randn.(rng)
function Base.randn(
    rng::AbstractVector{<:AbstractRNG},
    T,
    sz::Int...)
    return cat(randn.(rng, T, first(sz))...; dims=2)
end

# Notations
# ℓπ: log density of the target distribution
# ∂ℓπ∂θ: gradient of the log density of the target distribution
# θ: position variables / model parameters
# r: momentum variables

include("adaptation/Adaptation.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
export NesterovDualAveraging, Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor
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

# AD utilities
abstract type AbstractAD end
const AD_AMBIGUITY_ERROR_MSG = "MethodError: Hamiltonian(metric::AbstractMetric, ℓπ) is ambiguous because both Zygote and ForwardDiff is loaded. Please use Hamiltonian(metric, ℓπ, ZygoteAD) or Hamiltonian(metric, ℓπ, ForwardDiffAD) explictly."
function Hamiltonian(metric, ℓπ)
    error("MethodError: no method matching Hamiltonian(metric::AbstractMetric, ℓπ) because no AD backend is loaded. Please load Zygote or ForwardDiff before calling Hamiltonian(metric, ℓπ).")
end

using Requires
include("init.jl")

end # module
