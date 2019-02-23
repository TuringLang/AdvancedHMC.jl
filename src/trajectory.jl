abstract type AbstractTrajectory end

struct StaticTrajectory <: AbstractTrajectory
    integrator  ::  AbstractIntegrator
    n_points    ::  Integer
end

function points(st::StaticTrajectory)

end

function lastpoint(st::StaticTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    θ, r = step(st.integrator, h, θ, r, st.n_points)
end

struct NoUTurnTrajectory
    fields
end
