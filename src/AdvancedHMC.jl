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

### AD utilities

abstract type AbstractAD end

struct NoAD <: AbstractAD end

ADBACKEND = NoAD()

Hamiltonian(metric::AbstractMetric, ℓπ) = Hamiltonian(metric, ℓπ, ADBACKEND)

struct MultipleAD <: AbstractAD
    ads::Set{AbstractAD}
end

function update_ADBACKEND!(ad::AbstractAD)
    global ADBACKEND
    if ADBACKEND isa NoAD
        ADBACKEND = ad
    elseif ADBACKEND isa MultipleAD
        ADBACKEND = MultipleAD(Set((ADBACKEND.ads..., ad)))
    else
        ADBACKEND = MultipleAD(Set((ADBACKEND, ad)))
    end
end

function Hamiltonian(metric::AbstractMetric, ℓπ, ::NoAD)
    error("MethodError: no method matching Hamiltonian(metric::AbstractMetric, ℓπ) because no AD backend is loaded. Please load an AD package before calling Hamiltonian(metric, ℓπ).")
end

function Hamiltonian(metric::AbstractMetric, ℓπ, ad::MultipleAD)
    backend_list_str = join(" and ", string.(backend.(ad.ads)))
    constructors_str = join(" or ", map(s -> "Hamiltonian(metric, ℓπ, $s)", string.(ad.ads)))
    error("MethodError: Hamiltonian(metric::AbstractMetric, ℓπ) is ambiguous because $backend_list_str are available in the same time. Please use $constructors_str explictly.")
end

using Requires
include("init.jl")

end # module
