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
    return dpc.ve.n > 20 ? h(getM⁻¹(dpc)) : h, prop
end

function update(h::Hamiltonian, prop::AbstractProposal, dpc::DensePreConditioner)
    return dpc.ce.n > 20 ? h(getM⁻¹(dpc)) : h, prop
end
