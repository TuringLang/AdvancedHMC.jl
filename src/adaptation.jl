include("adaptation/Adaptation.jl")

using .Adaptation

function update(h::Hamiltonian, prop::AbstractProposal, da::DualAveraging)
    return h, prop(getss(da))
end
