####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

abstract type AbstractLeapfrog{T} <: AbstractIntegrator end

jitter(::AbstractRNG, ::AbstractLeapfrog, ϵ) = ϵ
temper(lf::AbstractLeapfrog, r, ::NamedTuple{(:i, :is_half),Tuple{Int64,Bool}}, ::Int) = r

function step(
    rng::AbstractRNG,
    lf::AbstractLeapfrog{T},
    h::Hamiltonian,
    z::PhasePoint,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0   # simulate hamiltonian backward when n_steps < 0,
) where {T<:AbstractFloat}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases
    ϵ = fwd ? lf.ϵ : -lf.ϵ
    ϵ = jitter(rng, lf, ϵ)

    @unpack θ, r = z
    @unpack value, gradient = ∂H∂θ(h, θ)
    for i = 1:n_steps
        # Tempering
        r = temper(lf, r, (i=i, is_half=true), n_steps)
        # Take a half leapfrog step for momentum variable
        r = r - ϵ / 2 * gradient
        # Take a full leapfrog step for position variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ * ∇r
        # Take a half leapfrog step for momentum variable
        @unpack value, gradient = ∂H∂θ(h, θ)
        r = r - ϵ / 2 * gradient
        # Tempering
        r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        !isfinite(z) && break
    end
    return z
end

function step(lf::AbstractLeapfrog, h::Hamiltonian, z::PhasePoint, n_steps::Int=1; fwd::Bool=n_steps > 0)
    return step(GLOBAL_RNG, lf, h, z, n_steps; fwd=fwd)
end

struct Leapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
end
Base.show(io::IO, l::Leapfrog) = print(io, "Leapfrog(ϵ=$(round(l.ϵ; sigdigits=3)))")

### Jittering

struct JitteredLeapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
    jitter  ::  T
end

function Base.show(io::IO, l::JitteredLeapfrog)
    print(io, "JitteredLeapfrog(ϵ=$(round(l.ϵ; sigdigits=3)), jitter=$(round(l.jitter; sigdigits=3)))")
end

# Jitter step size; ref: https://github.com/stan-dev/stan/blob/1bb054027b01326e66ec610e95ef9b2a60aa6bec/src/stan/mcmc/hmc/base_hmc.hpp#L177-L178
jitter(rng::AbstractRNG, lf::JitteredLeapfrog, ϵ) = ϵ * (1 + lf.jitter * (2 * rand(rng) - 1))

### Tempering

struct TemperedLeapfrog{T<:AbstractFloat} <: AbstractLeapfrog{T}
    ϵ       ::  T
    α       ::  T
end

function Base.show(io::IO, l::TemperedLeapfrog)
    print(io, "TemperedLeapfrog(ϵ=$(round(l.ϵ; sigdigits=3)), α=$(round(l.α; sigdigits=3)))")
end

"""
    temper(lf::TemperedLeapfrog, r, step::NamedTuple{(:i, :is_half),Tuple{Int64,Bool}}, n_steps::Int)

Tempering step. `step` is a named tuple with 
- `i` being the current leapfrog iteration and 
- `is_half` indicating whether or not it's (the first) half momentum/tempering step
"""
function temper(lf::TemperedLeapfrog, r, step::NamedTuple{(:i, :is_half),Tuple{Int64,Bool}}, n_steps::Int)
    i_temper = 2(step.i - 1) + 1 + step.is_half    # counter for half temper steps
    return i_temper <= n_steps ? r * sqrt(lf.α) : r / sqrt(lf.α)
end
