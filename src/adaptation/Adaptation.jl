module Adaptation
export Adaptation

using LinearAlgebra: LinearAlgebra
using Statistics: Statistics

using ..AdvancedHMC: AbstractScalarOrVec
using DocStringExtensions

"""
$(TYPEDEF)

Abstract type for HMC adaptors. 
"""
abstract type AbstractAdaptor end
function getM⁻¹ end
function getϵ end
function adapt! end
function reset! end
function initialize! end
function finalize! end
export AbstractAdaptor, adapt!, initialize!, finalize!, reset!, getϵ, getM⁻¹

struct NoAdaptation <: AbstractAdaptor end
export NoAdaptation
include("stepsize.jl")
export StepSizeAdaptor, NesterovDualAveraging
include("massmatrix.jl")
export MassMatrixAdaptor, UnitMassMatrix, WelfordVar, WelfordCov

##
## Composite adaptors
## TODO: generalise this to a list of adaptors
##

struct NaiveHMCAdaptor{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc::M
    ssa::Tssa
end
function Base.show(io::IO, ::MIME"text/plain", a::NaiveHMCAdaptor)
    return print(io, "NaiveHMCAdaptor(pc=", a.pc, ", ssa=", a.ssa, ")")
end

getM⁻¹(ca::NaiveHMCAdaptor) = getM⁻¹(ca.pc)
getϵ(ca::NaiveHMCAdaptor) = getϵ(ca.ssa)

# TODO: implement consensus adaptor
function adapt!(
    nca::NaiveHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
)
    adapt!(nca.ssa, θ, α)
    adapt!(nca.pc, θ, α)
    return nothing
end
function reset!(aca::NaiveHMCAdaptor)
    reset!(aca.ssa)
    reset!(aca.pc)
    return nothing
end
initialize!(adaptor::NaiveHMCAdaptor, n_adapts::Int) = nothing
finalize!(aca::NaiveHMCAdaptor) = finalize!(aca.ssa)

include("stan_adaptor.jl")
export NaiveHMCAdaptor, StanHMCAdaptor

end # module
