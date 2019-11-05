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

# Notations
# ℓπ: log density of the target distribution
# ∂ℓπ∂θ: gradient of the log density of the target distribution
# θ: position variables / model parameters
# r: momentum variables

include("Adaptation/Adaptation.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
export NesterovDualAveraging, Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor
using .Adaptation
import .Adaptation: adapt!

include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog
include("Transition/Transition.jl")
export HMC, HMCDA, NUTS, SliceTS, MultinomialTS, ClassicNoUTurn, GeneralisedNoUTurn, find_good_eps
export SpiralHMC

include("diagnosis.jl")
include("sampler.jl")
export sample

end # module
