module Adaptation

import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG

abstract type AbstractAdapter end

include("stepsize.jl")
include("precond.jl")

abstract type CompositeAdapter <: AbstractAdapter end

struct NaiveCompAdapter <: CompositeAdapter
    pc  :: AbstractPreConditioner
    ssa :: StepSizeAdapter
end

function getM⁻¹(ca::CompositeAdapter)
    return getM⁻¹(ca.pc)
end

function getss(ca::CompositeAdapter)
    return getss(ca.ssa)
end

export adapt!,
       DualAveraging, getss,
       UnitPreConditioner, DiagPreConditioner, DensePreConditioner, getM⁻¹

end # module
