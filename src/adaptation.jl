include("adaptation/Adaptation.jl")

using .Adaptation

function PreConditioner(::UnitEuclideanMetric)
    return UnitPreConditioner()
end

function PreConditioner(m::DiagEuclideanMetric)
    return DiagPreConditioner(m.dim)
end

function PreConditioner(m::DenseEuclideanMetric)
    return DensePreConditioner(m.dim)
end

function update(h::Hamiltonian, prop::AbstractProposal, dpc::Adaptation.AbstractPreConditioner)
    return h(getM⁻¹(dpc)), prop
end

function update(h::Hamiltonian, prop::AbstractProposal, da::NesterovDualAveraging)
    return h, prop(prop.integrator(getϵ(da)))
end

function update(h::Hamiltonian, prop::AbstractProposal, ca::Adaptation.AbstractCompositeAdaptor)
    return h(getM⁻¹(ca.pc)), prop(prop.integrator(getϵ(ca.ssa)))
end
