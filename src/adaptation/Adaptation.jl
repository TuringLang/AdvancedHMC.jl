module Adaptation

import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG

abstract type AbstractAdapter end

include("stepsize.jl")
include("precond.jl")

export adapt!,
       DualAveraging, getss,
       UnitPreConditioner, DiagPreConditioner, DensePreConditioner, getM⁻¹

end # module
