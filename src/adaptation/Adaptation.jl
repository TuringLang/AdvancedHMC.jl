module Adaptation

import Base: string
import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG

abstract type AbstractAdapter end

include("stepsize.jl")
include("precond.jl")

abstract type AbstractCompositeAdapter <: AbstractAdapter end

# TODO: generalise this to a list of adapters
struct NaiveCompAdapter <: AbstractCompositeAdapter
    pc  :: AbstractPreConditioner
    ssa :: StepSizeAdapter
end

function adapt!(tp::NaiveCompAdapter, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt!(tp.ssa, θ, α)
    adapt!(tp.pc, θ, α)
end

function getM⁻¹(ca::AbstractCompositeAdapter)
    return getM⁻¹(ca.pc)
end

function getss(ca::AbstractCompositeAdapter)
    return getss(ca.ssa)
end

include("threephase.jl")

export adapt!,
       DualAveraging, getss,
       UnitPreConditioner, DiagPreConditioner, DensePreConditioner, getM⁻¹,
       NaiveCompAdapter, ThreePhaseAdapter

end # module
