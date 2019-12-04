module Adaptation

import Base: string, rand
using Random: GLOBAL_RNG, AbstractRNG
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky, UniformScaling
import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG
using Parameters: @unpack, @pack!

const AbstractScalarOrVec{T} = Union{T,AbstractVector{T}} where {T<:AbstractFloat}

abstract type AbstractAdaptor end

##
## Interface for adaptors
##

getM⁻¹(adaptor::T) where {T<:AbstractAdaptor} = error("`getM⁻¹(adaptor::$T)` is not implemented.")
getϵ(adaptor::T) where {T<:AbstractAdaptor} = error("`getϵ(adaptor::$T)` is not implemented.")
adapt!(
    adaptor::T,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
) where {T<:AbstractAdaptor} = error("`adapt!(adaptor::$T, θ::AbstractVecOrMat{<:AbstractFloat}, α::AbstractScalarOrVec{<:AbstractFloat})` is not implemented.")
reset!(adaptor::T) where {T<:AbstractAdaptor} = error("`reset!(adaptor::$T)` is not implemented.")
initialize!(adaptor::T, n_adapts::Int) where {T<:AbstractAdaptor} = throw(MethodError(initialize!, adaptor, n_adapts))
finalize!(adaptor::T) where {T<:AbstractAdaptor} = error("`finalize!(adaptor::$T)` is not implemented.")

struct NoAdaptation <: AbstractAdaptor end

include("stepsize.jl")
include("precond.jl")

# TODO: implement consensus adaptor

##
## Compositional adaptor
## TODO: generalise this to a list of adaptors
##

struct NaiveHMCAdaptor{M<:AbstractPreconditioner, Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc  :: M
    ssa :: Tssa
end
Base.show(io::IO, a::NaiveHMCAdaptor) = print(io, "NaiveHMCAdaptor(pc=$(a.pc), ssa=$(a.ssa))")

getM⁻¹(aca::NaiveHMCAdaptor) = getM⁻¹(aca.pc)
getϵ(aca::NaiveHMCAdaptor) = getϵ(aca.ssa)
function adapt!(
    nca::NaiveHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
)
    adapt!(nca.ssa, θ, α)
    adapt!(nca.pc, θ, α)
end
function reset!(aca::NaiveHMCAdaptor)
    reset!(aca.ssa)
    reset!(aca.pc)
end
initialize!(adaptor::NaiveHMCAdaptor, n_adapts::Int) = nothing
finalize!(aca::NaiveHMCAdaptor) = finalize!(aca.ssa)

##
## Stan's windowed adaptor.
##
include("stan_adaption.jl")

export adapt!, initialize!, finalize!, getϵ, getM⁻¹, reset!, renew,
       NesterovDualAveraging,
       UnitPreconditioner, DiagPreconditioner, DensePreconditioner,
       AbstractMetric, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric,
       Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor

end # module
