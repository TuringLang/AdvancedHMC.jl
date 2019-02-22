abstract type AbstractTrajectorySampler end
struct LastFromTraj <: AbstractTrajectorySampler end
struct UniformFromTraj <: AbstractTrajectorySampler end
struct MultinomialFromTraj <: AbstractTrajectorySampler end

abstract type AbstractTrajectory end

struct StaticTrajectory{S<:AbstractTrajectorySampler}
    sampler     ::  S
    integrator  ::  AbstractIntegrator
    n_points    ::  Integer
end

function build_and_sample(st::StaticTrajectory{LastFromTraj}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T})
    θ, r = step(st.integrator, h, θ, r, st.n_points)
    return θ, r
end

struct NoUTurnTrajectory
    fields
end
