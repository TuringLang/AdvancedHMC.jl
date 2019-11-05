abstract type StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed leapfrog step numbers.
###
struct HMC{I<:AbstractIntegrator} <: StaticTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end
Base.show(io::IO, τ::HMC) = print(io, "HMC(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

function transition(
    rng::AbstractRNG,
    τ::HMC,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    z′ = step(rng, τ.integrator, h, z, τ.n_steps)
    # Accept via MH criteria
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    if is_accept
        # Reverse momentum variable to preserve reversibility
        z = PhasePoint(z′.θ, -z′.r, z′.ℓπ, z′.ℓκ)
    end
    stat = (
        step_size=τ.integrator.ϵ,
        n_steps=τ.n_steps,
        is_accept=is_accept,
        acceptance_rate=α,
        log_density=z.ℓπ.value,
        hamiltonian_energy=energy(z),
       )
    return Transition(z, stat)
end

###
### Spiral HMC
###
struct SpiralHMC{I<:AbstractIntegrator} <: StaticTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end
Base.show(io::IO, τ::SpiralHMC) = print(io, "SpiralHMC(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

function transition(
    rng::AbstractRNG,
    τ::SpiralHMC,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    z′ = step(rng, τ.integrator, h, z, τ.n_steps)
    # Accept via MH criteria
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    if is_accept
        # Reverse momentum variable to preserve reversibility
        z = PhasePoint(z′.θ, -z′.r, z′.ℓπ, z′.ℓκ)
    end
    stat = (
        step_size=τ.integrator.ϵ,
        n_steps=τ.n_steps,
        is_accept=is_accept,
        acceptance_rate=α,
        log_density=z.ℓπ.value,
        hamiltonian_energy=energy(z),
       )
    return Transition(z, stat)
end
