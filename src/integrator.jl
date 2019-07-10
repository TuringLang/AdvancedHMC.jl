####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ   ::  T
end

# Create a `Leapfrog` with a new `ϵ`
function (::Leapfrog)(ϵ::AbstractFloat)
    return Leapfrog(ϵ)
end

# TODO: double check the function below to see if it is type stable or not
function step(
    lf::Leapfrog{F},
    h::Hamiltonian,
    z::PhasePoint,
    n_steps::Int=1;
    fwd = n_steps > 0 # Simulate hamiltonian backward when n_steps < 0
) where {F<:AbstractFloat,T<:Real}
    @unpack θ, r = z
    ϵ = fwd ? lf.ϵ : -lf.ϵ

    @unpack value, gradient = ∂H∂θ(h, θ)
    for i = 1:abs(n_steps)
        r = r - ϵ/2 * gradient # Take a half leapfrog step for momentum variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ * ∇r   # Take a full leapfrog step for position variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ/2 * gradient # Take a half leapfrog step for momentum variable
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        !isfinite(z) && break
    end
    return z
end
