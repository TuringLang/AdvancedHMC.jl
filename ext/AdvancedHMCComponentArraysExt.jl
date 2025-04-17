module AdvancedHMCComponentArraysExt

using LinearAlgebra

using AdvancedHMC:
    AdvancedHMC,
    Hamiltonian,
    UnitEuclideanMetric,
    DiagEuclideanMetric,
    DenseEuclideanMetric,
    GaussianKinetic
using ComponentArrays: ComponentVecOrMat, ComponentVector, ComponentMatrix, getaxes

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:DiagEuclideanMetric,<:GaussianKinetic}, r::ComponentVecOrMat
)
    (; M⁻¹) = h.metric
    (getaxes(M⁻¹) !== getaxes(r)) &&
        throw(ArgumentError("Axes of mass matrix and momentum must match"))
    return h.metric.M⁻¹ .* r
end
function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:DenseEuclideanMetric,<:GaussianKinetic}, r::ComponentVector
)
    (; M⁻¹) = h.metric
    (last(getaxes(M⁻¹)) !== first(getaxes(r))) &&
        throw(ArgumentError("Axes of mass matrix and momentum must match"))
    return h.metric.M⁻¹ * r
end
function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:DenseEuclideanMetric,<:GaussianKinetic}, r::ComponentMatrix
)
    (; M⁻¹) = h.metric
    getaxes(M⁻¹) !== getaxes(r) &&
        throw(ArgumentError("Axes of mass matrix and momentum must match"))
    return h.metric.M⁻¹ * r
end

end # module
