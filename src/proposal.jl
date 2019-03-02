abstract type AbstractProposal{T<:AbstractTrajectory} end

struct TakeLastProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end

function propose(tlp::TakeLastProposal{TJ}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {TJ<:StaticTrajectory,T<:Real}
    θ, r, _ = lastpoint(tlp.traj, h, θ, r)
    return θ, -r
end

function propose(rng::AbstractRNG, tlp::TakeLastProposal{TJ}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {TJ<:NoUTurnTrajectory,T<:Real}
    θ, r, _ = lastpoint(rng, tlp.traj, h, θ, r)
    return θ, -r
end

function propose(tlp::TakeLastProposal{TJ}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {TJ<:NoUTurnTrajectory,T<:Real}
    return propose(GLOBAL_RNG, tlp, h, θ, r)
end

struct UniformProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end

struct MultinomialProposal{T<:AbstractTrajectory} <: AbstractProposal{T}
    traj    ::  T
end
