module AdvancedHMC

const DEBUG = convert(Bool, parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
using StatsFuns: logaddexp, logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter: ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck

import StatsBase: sample
import Parameters: reconstruct

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
@deprecate find_good_eps find_good_stepsize
export EndPointTS, SliceTS, MultinomialTS, 
       StaticTrajectory, HMCDA, NUTS, 
       ClassicNoUTurn, GeneralisedNoUTurn, 
       find_good_stepsize

include("adaptation/Adaptation.jl")
using .Adaptation
import .Adaptation: StepSizeAdaptor, MassMatrixAdaptor, StanHMCAdaptor, NesterovDualAveraging

# Helpers for initializing adaptors via AHMC structs

StepSizeAdaptor(δ::AbstractFloat, i::AbstractIntegrator) =
    NesterovDualAveraging(δ, nom_step_size(i))

MassMatrixAdaptor(m::UnitEuclideanMetric{T}) where {T} =
    UnitMassMatrix{T}()
MassMatrixAdaptor(m::DiagEuclideanMetric{T}) where {T} =
    WelfordVar{T}(size(m); var=copy(m.M⁻¹))
MassMatrixAdaptor(m::DenseEuclideanMetric{T}) where {T} =
    WelfordCov{T}(size(m); cov=copy(m.M⁻¹))

MassMatrixAdaptor(
    m::Type{TM},
    sz::Tuple{Vararg{Int}}=(2,)
) where {TM<:AbstractMetric} = MassMatrixAdaptor(Float64, m, sz)

MassMatrixAdaptor(
    ::Type{T},
    ::Type{TM},
    sz::Tuple{Vararg{Int}}=(2,)
) where {T, TM<:AbstractMetric} = MassMatrixAdaptor(TM(T, sz))

# Deprecations

@deprecate StanHMCAdaptor(n_adapts, pc, ssa) initialize!(StanHMCAdaptor(pc, ssa), n_adapts)
@deprecate NesterovDualAveraging(δ, i::AbstractIntegrator) StepSizeAdaptor(δ, i)
@deprecate Preconditioner(m::AbstractMatrix) MassMatrixAdaptor(m)

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
