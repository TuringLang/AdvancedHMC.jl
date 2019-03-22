module Adaptation

using ..AdvancedHMC: DEBUG

abstract type AbstractAdapter end

include("stepsize.jl")

export DualAveraging, getss, adapt!

end # module
