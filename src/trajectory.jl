abstract type AbstractTrajectory end

struct StaticTrajectory <: AbstractTrajectory
    integrator  ::  AbstractIntegrator
    n_points    ::  Integer
end

function points(st::StaticTrajectory)
    return [step(st.integrator, h, θ, r) for i = 1:st.n_points]
end

function lastpoint(st::StaticTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return steps(st.integrator, h, θ, r, st.n_points)
end

struct NoUTurnTrajectory
    fields
end
