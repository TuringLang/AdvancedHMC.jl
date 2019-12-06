abstract type StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed leapfrog step numbers.
###

struct HMC{S<:AbstractTrajectorySampler,I<:AbstractIntegrator} <: StaticTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end

Base.show(io::IO, τ::HMC{S,I}) where {I,S<:LastTS} =
    print(io, "HMC{LastTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")
Base.show(io::IO, τ::HMC{S,I}) where {I,S<:MultinomialTS} =
    print(io, "HMC{MultinomialTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

HMC{S}(integrator::I, n_steps::Int) where {S,I} = HMC{S,I}(integrator, n_steps)
HMC(args...) = HMC{LastTS}(args...) # default HMC using last point from trajectory

function transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    τ::HMC,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    H0 = energy(z)
    integrator = jitter(rng, τ.integrator)
    z′ = step(τ.integrator, h, z, τ.n_steps)
    # Are we going to accept the `z′` via MH criteria?
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    # Do the actual accept / reject
    z = accept_phasepoint!(z, z′, is_accept)    # NOTE: this function changes `z′` in place in matrix-parallel mode
    # Reverse momentum variable to preserve reversibility
    z = PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    H = energy(z)
    tstat = merge(
        (
         n_steps=τ.n_steps,
         is_accept=is_accept,
         acceptance_rate=α,
         log_density=z.ℓπ.value,
         hamiltonian_energy=H,
         hamiltonian_energy_error=H - H0
        ),
        stat(integrator)
    )
    return Transition(z, tstat)
end

# Return the accepted phase point
function accept_phasepoint!(z::T, z′::T, is_accept::Bool) where {T<:PhasePoint{<:AbstractVector}}
    if is_accept
        return z′
    else
        return z
    end
end
function accept_phasepoint!(z::T, z′::T, is_accept) where {T<:PhasePoint{<:AbstractMatrix}}
    # Revert unaccepted proposals in `z′`
    if any((!).(is_accept))
        z′.θ[:,(!).(is_accept)] = z.θ[:,(!).(is_accept)]
        z′.r[:,(!).(is_accept)] = z.r[:,(!).(is_accept)]
        z′.ℓπ.value[(!).(is_accept)] = z.ℓπ.value[(!).(is_accept)]
        z′.ℓπ.gradient[:,(!).(is_accept)] = z.ℓπ.gradient[:,(!).(is_accept)]
        z′.ℓκ.value[(!).(is_accept)] = z.ℓκ.value[(!).(is_accept)]
        z′.ℓκ.gradient[:,(!).(is_accept)] = z.ℓκ.gradient[:,(!).(is_accept)]
    end
    # Always return `z′` as any unaccepted proposal is already reverted
    # NOTE: This in place treatment of `z′` is for memory efficient consideration.
    #       We can also copy `z′ and avoid mutating the original `z′`. But this is
    #       not efficient and immutability of `z′` is not important in this local scope.
    return z′
end
