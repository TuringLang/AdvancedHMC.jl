module AdvancedHMC

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
using StatsFuns: logaddexp, logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter: ProgressMeter
using Parameters: Parameters, @unpack, reconstruct
using ArgCheck: @argcheck

import StatsBase: sample

const AbstractScalarOrVec{T} = Union{T,AbstractVector{T}} where {T<:AbstractFloat}

include("utilities.jl")

# Notations
# ℓπ: log density of the target distribution
# θ: position variables / model parameters
# ∂ℓπ∂θ: gradient of the log density of the target distribution w.r.t θ
# r: momentum variables
# z: phase point / a pair of θ and r

include("metric.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric

include("hamiltonian.jl")
export Hamiltonian

include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog

include("trajectory.jl")
export EndPointTS, SliceTS, MultinomialTS, 
       StaticTrajectory, HMCDA, NUTS, 
       ClassicNoUTurn, GeneralisedNoUTurn, 
       find_good_eps

include("adaptation/Adaptation.jl")
using .Adaptation
export StepSizeAdaptor, NesterovDualAveraging, 
       MassMatrixAdaptor, UnitMassMatrix, WelfordVar, WelfordCov, 
       NaiveHMCAdaptor, StanHMCAdaptor

include("diagnosis.jl")

include("sampler.jl")
export sample

### Init

using Requires

function __init__()
    include(joinpath(@__DIR__, "contrib", "diffeq.jl"))
    include(joinpath(@__DIR__, "contrib", "ad.jl"))
end

end # module
