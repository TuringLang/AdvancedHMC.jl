module Adaptation

import Base: string, rand
using Random: GLOBAL_RNG, AbstractRNG
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG

abstract type AbstractAdaptor end
finalize!(::AbstractAdaptor) = nothing
struct NoAdaptation <: AbstractAdaptor end

include("stepsize.jl")
include("precond.jl")

abstract type AbstractCompositeAdaptor <: AbstractAdaptor end

# TODO: generalise this to a list of adaptors
struct NaiveCompAdaptor <: AbstractCompositeAdaptor
    pc  :: AbstractPreconditioner
    ssa :: StepSizeAdaptor
end

function adapt!(nca::NaiveCompAdaptor, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt!(nca.ssa, θ, α)
    adapt!(nca.pc, θ, α)
end

function getM⁻¹(aca::AbstractCompositeAdaptor)
    return getM⁻¹(aca.pc)
end

function getϵ(aca::AbstractCompositeAdaptor)
    return getϵ(aca.ssa)
end

function finalize!(aca::AbstractCompositeAdaptor)
    finalize!(aca.ssa)
end

include("stan_adaption.jl")

export adapt!, finalize!, getϵ, getM⁻¹, 
       NesterovDualAveraging,
       UnitPreconditioner, DiagPreconditioner, DensePreconditioner,
       AbstractMetric, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric,
       Preconditioner, NaiveCompAdaptor, StanNUTSAdaptor

end # module
