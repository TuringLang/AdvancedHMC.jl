module Adaptation

import Base: string, rand
using Random: GLOBAL_RNG, AbstractRNG
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
import LinearAlgebra, Statistics
using ..AdvancedHMC: DEBUG
using Parameters: @unpack, @pack!

abstract type AbstractAdaptor end

##
## Interface for adaptors
##

adapt!(
    ::AbstractAdaptor,
    ::AbstractVector{T},
    ::AbstractFloat,
    is_update::Bool=true
) where {T<:Real}       = nothing
getM⁻¹(::AbstractAdaptor) = nothing
getϵ(::AbstractAdaptor)   = nothing
reset!(::AbstractAdaptor) = nothing
finalize!(::AbstractAdaptor) = nothing

struct NoAdaptation <: AbstractAdaptor end

include("stepsize.jl")
include("precond.jl")


##
## Compositional adaptor
## TODO: generalise this to a list of adaptors
##

struct NaiveHMCAdaptor{M<:AbstractPreconditioner, Tssa <: StepSizeAdaptor} <: AbstractAdaptor
    pc  :: M
    ssa :: Tssa
end

Base.show(io::IO, a::NaiveHMCAdaptor) = print(io, "NaiveHMCAdaptor(pc=$(a.pc), ssa=$(a.ssa))")

getM⁻¹(aca::NaiveHMCAdaptor) = getM⁻¹(aca.pc)
getϵ(aca::NaiveHMCAdaptor)   = getϵ(aca.ssa)
finalize!(aca::NaiveHMCAdaptor) = finalize!(aca.ssa)
function adapt!(nca::NaiveHMCAdaptor, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt!(nca.ssa, θ, α)
    adapt!(nca.pc, θ, α)
end
function reset!(aca::NaiveHMCAdaptor)
    reset!(aca.ssa)
    reset!(aca.pc)
end

##
## Stan's windowed adaptor.
##
include("stan_adaption.jl")

export adapt!, finalize!, getϵ, getM⁻¹, reset!,
       NesterovDualAveraging,
       UnitPreconditioner, DiagPreconditioner, DensePreconditioner,
       AbstractMetric, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric,
       Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor

end # module
