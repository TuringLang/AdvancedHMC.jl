module Adaptation
export Adaptation

using LinearAlgebra: LinearAlgebra
using Statistics: Statistics
using Parameters: @unpack, @pack!

using ..AdvancedHMC: DEBUG, AbstractScalarOrVec

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
include("composite.jl")
export CompositeAdaptor, NaiveHMCAdaptor, StanHMCAdaptor

# User helpers for AdvancedHMC

using ..AdvancedHMC: 
    AbstractIntegrator, nom_step_size,
    AbstractMetric, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric

StepSizeAdaptor(δ::AbstractFloat, i::AbstractIntegrator) = 
    NesterovDualAveraging(δ, nom_step_size(i))

@deprecate NesterovDualAveraging(δ, i::AbstractIntegrator) StepSizeAdaptor(δ, i)

MassMatrixAdaptor(m::UnitEuclideanMetric{T}) where {T} = 
    UnitMassMatrix(T)
MassMatrixAdaptor(m::DiagEuclideanMetric{T}) where {T} = 
    WelfordVar(T, size(m); var=copy(m.M⁻¹))
MassMatrixAdaptor(m::DenseEuclideanMetric{T}) where {T} = 
    WelfordCov(T, size(m); cov=copy(m.M⁻¹))

@deprecate Preconditioner(m::AbstractMatrix) MassMatrixAdaptor(m)

MassMatrixAdaptor(
    m::Type{TM}, 
    sz::Tuple{Vararg{Int}}=(2,)
) where {TM<:AbstractMetric} = MassMatrixAdaptor(Float64, m, sz)

MassMatrixAdaptor(
    ::Type{T}, 
    ::Type{TM}, 
    sz::Tuple{Vararg{Int}}=(2,)
) where {T, TM<:AbstractMetric} = MassMatrixAdaptor(TM(T, sz))

end # module
