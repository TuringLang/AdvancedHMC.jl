####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ       ::  T
    jitter  :: T
end
Base.show(io::IO, l::Leapfrog) = print(io, "Leapfrog(ϵ=$(round(l.ϵ; sigdigits=3)))")

Leapfrog(ϵ::T) where {T<:AbstractFloat} = Leapfrog(ϵ, zero(T))

# TODO: double check the function below to see if it is type stable or not
function step(
    lf::Leapfrog{F},
    h::Hamiltonian,
    z::PhasePoint,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0 # simulate hamiltonian backward when n_steps < 0,
) where {F<:AbstractFloat,T<:Real}
    @unpack θ, r = z
    ϵ = fwd ? lf.ϵ : -lf.ϵ
    # Jitter step size; ref: https://github.com/stan-dev/stan/blob/1bb054027b01326e66ec610e95ef9b2a60aa6bec/src/stan/mcmc/hmc/base_hmc.hpp#L177-L178
    iszero(lf.jitter) || (ϵ *= (1 + lf.jitter * (2 * rand() - 1)))

    @unpack value, gradient = ∂H∂θ(h, θ)
    for i = 1:abs(n_steps)
        r = r - ϵ / 2 * gradient # take a half leapfrog step for momentum variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ * ∇r   # take a full leapfrog step for position variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 * gradient # take a half leapfrog step for momentum variable
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        !isfinite(z) && break
    end
    return z
end
