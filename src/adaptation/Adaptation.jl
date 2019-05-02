module Adaptation

import Base: string
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG

abstract type AbstractAdaptor end

include("stepsize.jl")
include("precond.jl")

abstract type AbstractCompositeAdaptor <: AbstractAdaptor end

# TODO: generalise this to a list of adaptors
struct NaiveCompAdaptor <: AbstractCompositeAdaptor
    pc  :: AbstractPreConditioner
    ssa :: StepSizeAdaptor
end

function adapt!(tp::NaiveCompAdaptor, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt!(tp.ssa, θ, α)
    adapt!(tp.pc, θ, α)
end

function getM⁻¹(ca::AbstractCompositeAdaptor)
    return getM⁻¹(ca.pc)
end

function getϵ(ca::AbstractCompositeAdaptor)
    return getϵ(ca.ssa)
end

include("stan_adaption.jl")

export adapt!, getϵ, getM⁻¹,
       NesterovDualAveraging,
       UnitPreConditioner, DiagPreConditioner, DensePreConditioner,
       AbstractMetric, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric,
       PreConditioner, NaiveCompAdaptor, StanNUTSAdaptor

end # module
