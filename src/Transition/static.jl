abstract type StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed leapfrog step numbers.
###
struct HMC{S<:AbstractTrajectorySampler,I<:AbstractIntegrator} <: StaticTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end
Base.show(io::IO, τ::HMC) = print(io, "HMC(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

HMC{S}(integrator::I, n_steps::Int) where {S,I} = HMC{S,I}(integrator, n_steps)
HMC(args...) = HMC{LastTS}(args...) # default HMC using last point from trajectory

function transition(
    rng::AbstractRNG,
    τ::HMC,
    h::Hamiltonian,
    z::PhasePoint
)
    z′ = samplecand(rng, τ, h, z)
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

### Last from trajecory

samplecand(rng, τ::HMC{LastTS}, h, z) = step(rng, τ.integrator, h, z, τ.n_steps)

### Multinomial sampling from trajecory

function randcat(rng::AbstractRNG, xs, p)
    u = rand(rng)
    cp = zero(eltype(p))
    i = 0
    while cp < u
        cp += p[i +=1]
    end
    return xs[max(i, 1)]
end

function samplecand(rng, τ::HMC{MultinomialTS}, h, z)
    zs = steps(rng, τ.integrator, h, z, τ.n_steps)
    ℓws = -energy.(zs)
    ℓws = ℓws .= maximum(ℓws)
    p_unorm = exp.(ℓws)
    return randcat(rng, zs, p_unorm / sum(p_unorm))
end
