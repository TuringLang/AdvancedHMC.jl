abstract type AbstractTrajectory end

struct StaticTrajectory <: AbstractTrajectory
    integrator  ::  AbstractIntegrator
    n_steps    ::  Integer
end

function points(st::StaticTrajectory, h::Hamiltonian, θ::T, r::T) where {T<:AbstractVector{<:Real}}
    ps = Vector{Tuple{T,T}}(undef, st.n_steps + 1)
    ps[1] = (θ, r)
    for i = 2:st.n_steps+1
        ps[i] = step(st.integrator, h, ps[i]...)
    end
    return ps
end

function lastpoint(st::StaticTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return steps(st.integrator, h, θ, r, st.n_steps)
end

struct NoUTurnTrajectory
    fields
end
