# TODO: argument below with trajectory type
abstract type AbstractProposal end

struct TakeLastProposal{T<:AbstractTrajectory} <: AbstractProposal
    traj    ::  T
end

function propose(tlp::TakeLastProposal, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    θ, r = lastpoint(tlp.traj, h, θ, r)
    return θ, r
end

struct UniformProposal <: AbstractProposal
    fields
end

struct MultinomialProposal <: AbstractProposal
    fields
end
