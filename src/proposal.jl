abstract type AbstractProposal{T<:AbstractTrajectory} end

struct TakeLastProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end

function propose(tlp::TakeLastProposal, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    θ, r, _ = lastpoint(tlp.traj, h, θ, r)
    return θ, -r
end

struct UniformProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end

struct MultinomialProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end
