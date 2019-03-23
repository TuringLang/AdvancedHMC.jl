include("adaptation/Adaptation.jl")

using .Adaptation

function update(h::Hamiltonian, prop::AbstractProposal, da::DualAveraging)
    return h, prop(getss(da))
end

function PreConditioner(::UnitEuclideanMetric)
    return UnitPreConditioner()
end

function PreConditioner(m::DiagEuclideanMetric)
    return DiagPreConditioner(m.dim)
end

function PreConditioner(m::DenseEuclideanMetric)
    return DensePreConditioner(m.dim)
end

function update(h::Hamiltonian, prop::AbstractProposal, ::UnitPreConditioner)
    return h, prop
end

function update(h::Hamiltonian, prop::AbstractProposal, dpc::DiagPreConditioner)
    return h(getM⁻¹(dpc)), prop
end

function update(h::Hamiltonian, prop::AbstractProposal, dpc::DensePreConditioner)
    return h(getM⁻¹(dpc)), prop
end
