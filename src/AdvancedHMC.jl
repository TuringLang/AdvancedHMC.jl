module AdvancedHMC

const DEBUG = convert(Bool, parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
using StatsFuns: logaddexp, logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter: ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck

using DocStringExtensions: TYPEDEF, TYPEDFIELDS

import StatsBase: sample
import Parameters: reconstruct

include("utilities.jl")

# Notations
# θ: position variables / model parameters
# r: momentum variables
# z: phase point / a pair of θ and r
# θ₀: initial position
# r₀: initial momentum
# z₀: initial phase point
# ℓπ: log density of the target distribution
# ∇ℓπ: gradient of the log density of the target distribution w.r.t θ
# κ: kernel
# τ: trajectory
# ϵ: step size
# L: step number
# t: integration time

include("metric.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric

include("hamiltonian.jl")
export Hamiltonian

include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog

include("trajectory.jl")
@deprecate find_good_eps find_good_stepsize
export Trajectory, HMCKernel, MixtureKernel,
       FullRefreshment, PartialRefreshment,
       FixedNSteps, FixedLength, 
       ClassicNoUTurn, NoUTurn, StrictNoUTurn,
       MetropolisTS, SliceTS, MultinomialTS,
       find_good_stepsize

struct HMC{TS} end
HMC{TS}(int::AbstractIntegrator, L) where {TS} =
    HMCKernel(FullRefreshment(), Trajectory(int, FixedNSteps(L)), TS)
HMC(int::AbstractIntegrator, L) = HMC{MetropolisTS}(int, L)
HMC(ϵ::AbstractScalarOrVec{<:Real}, L) = HMC{MetropolisTS}(Leapfrog(ϵ), L)

struct StaticTrajectory{TS} end
@deprecate StaticTrajectory{TS}(args...) where {TS} HMC{TS}(args...)
@deprecate StaticTrajectory(args...) HMC(args...)

struct HMCDA{TS} end
HMCDA{TS}(int::AbstractIntegrator, λ) where {TS} =
    HMCKernel(FullRefreshment(), Trajectory(int, FixedLength(λ)), TS)
HMCDA(int::AbstractIntegrator, λ) = HMCDA{MetropolisTS}(int, λ)
HMCDA(ϵ::AbstractScalarOrVec{<:Real}, λ) = HMCDA{MetropolisTS}(Leapfrog(ϵ), λ)

struct NUTS{TS, TC} end
NUTS{TS, TC}(int::AbstractIntegrator) where {TS, TC} =
    HMCKernel(FullRefreshment(), Trajectory(int, TC()), TS)
NUTS(int::AbstractIntegrator) = NUTS{MultinomialTS, NoUTurn}(int)
NUTS(ϵ::AbstractScalarOrVec{<:Real}) = NUTS{MultinomialTS, NoUTurn}(Leapfrog(ϵ))

export HMC, StaticTrajectory, HMCDA, NUTS

include("adaptation/Adaptation.jl")
using .Adaptation
import .Adaptation: StepSizeAdaptor, MassMatrixAdaptor, StanHMCAdaptor, NesterovDualAveraging

# Helpers for initializing adaptors via AHMC structs

StepSizeAdaptor(δ::AbstractFloat, stepsize::AbstractScalarOrVec{<:AbstractFloat}) = 
    NesterovDualAveraging(δ, stepsize)
StepSizeAdaptor(δ::AbstractFloat, i::AbstractIntegrator) = StepSizeAdaptor(δ, nom_step_size(i))

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
@deprecate NesterovDualAveraging(δ::AbstractFloat, i::AbstractIntegrator) StepSizeAdaptor(δ, i)
@deprecate Preconditioner(args...) MassMatrixAdaptor(args...)

export StepSizeAdaptor, NesterovDualAveraging, 
       MassMatrixAdaptor, UnitMassMatrix, WelfordVar, WelfordCov, 
       NaiveHMCAdaptor, StanHMCAdaptor

include("diagnosis.jl")

include("sampler.jl")
export sample

include("contrib/ad.jl")

### Init

using Requires

function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        export DiffEqIntegrator
        include("contrib/diffeq.jl")
    end

    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
        include("contrib/forwarddiff.jl")
    end

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("contrib/zygote.jl")
    end
end

end # module
