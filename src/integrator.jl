####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ       ::  T
    jitter  ::  T
    α       ::  T
end
Base.show(io::IO, l::Leapfrog) = print(io, "Leapfrog(ϵ=$(round(l.ϵ; sigdigits=3)))")

Leapfrog(ϵ::T; jitter::T=zero(T), α::T=one(T)) where {T<:AbstractFloat} = Leapfrog{T}(ϵ, jitter, α)

function step(
    lf::Leapfrog{T},
    h::Hamiltonian,
    z::PhasePoint,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0 # simulate hamiltonian backward when n_steps < 0,
) where {T<:AbstractFloat}
    @unpack θ, r = z
    ϵ = fwd ? lf.ϵ : -lf.ϵ
    sqrtα = sqrt(lf.α)
    n_steps = abs(n_steps)
    # Jitter step size; ref: https://github.com/stan-dev/stan/blob/1bb054027b01326e66ec610e95ef9b2a60aa6bec/src/stan/mcmc/hmc/base_hmc.hpp#L177-L178
    iszero(lf.jitter) || (ϵ *= (1 + lf.jitter * (2 * rand() - 1)))

    @unpack value, gradient = ∂H∂θ(h, θ)
    for i = 1:n_steps
        # Tempering; `ceil` includes mid if `n_steps` is odd, e.g. `<= ceil(5 / 2)` => `<= 3` 
        r = i <= ceil(Int, n_steps / 2) ? r * sqrtα : r / sqrtα     
        r = r - ϵ / 2 * gradient    # take a half leapfrog step for momentum variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ * ∇r  # take a full leapfrog step for position variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 * gradient    # take a half leapfrog step for momentum variable
        # Tempering; `floor` excludes mid if `n_steps` is odd, e.g. `<= floor(5 / 2)` => `<= 2` 
        r = i <= floor(Int, n_steps / 2) ? r * sqrtα : r / sqrtα
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        !isfinite(z) && break
    end
    return z
end
