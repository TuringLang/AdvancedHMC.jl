module AdvancedHMCComponentArraysExt

using LinearAlgebra

if isdefined(Base, :get_extension)
    using AdvancedHMC: AdvancedHMC, Hamiltonian, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric, GaussianKinetic
    using ComponentArrays: ComponentArray
else
    import ..AdvancedHMC: AdvancedHMC, Hamiltonian, UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric, GaussianKinetic
    import ..ComponentArrays: ComponentArray
end

function AdvancedHMC.∂H∂r(h::Hamiltonian{<:UnitEuclideanMetric,<:GaussianKinetic}, r::ComponentArray)
    copy(r)
end
function AdvancedHMC.∂H∂r(h::Hamiltonian{<:DiagEuclideanMetric,<:GaussianKinetic}, r::ComponentArray)
    out = similar(r)
    out .= h.metric.M⁻¹ .* r
    return out
end
function AdvancedHMC.∂H∂r(h::Hamiltonian{<:DenseEuclideanMetric,<:GaussianKinetic}, r::ComponentArray)
    out = similar(r)
    mul!(out, h.metric.M⁻¹, r)
    return out
end

end # module
