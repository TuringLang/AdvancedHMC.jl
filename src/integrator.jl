####
#### Numerical methods for simulating Hamiltonian trajectory.
####

# TODO: The type `<:Tuple{Integer,Bool}` is introduced to address
# https://github.com/TuringLang/Turing.jl/pull/941#issuecomment-549191813
# We might want to simplify it to `Tuple{Int,Bool}` when we figured out
# why the it behaves unexpected on Windows 32.

"""
$(TYPEDEF)

Represents an integrator used to simulate the Hamiltonian system.

# Implementation
A `AbstractIntegrator` is expected to have the following implementations:
- `stat`(@ref)
- `nom_step_size`(@ref)
- `step_size`(@ref)
"""
abstract type AbstractIntegrator end

stat(::AbstractIntegrator) = NamedTuple()

"""
    nom_step_size(::AbstractIntegrator)

Get the nominal integration step size. The current integration step size may
differ from this, for example if the step size is jittered. Nominal step size is
usually used in adaptation.
"""
nom_step_size(i::AbstractIntegrator) = step_size(i)

"""
    step_size(::AbstractIntegrator)

Get the current integration step size.
"""
function step_size end

"""
    update_nom_step_size(i::AbstractIntegrator, ϵ) -> AbstractIntegrator

Return a copy of the integrator `i` with the new nominal step size ([`nom_step_size`](@ref))
`ϵ`.
"""
function update_nom_step_size end

abstract type AbstractLeapfrog{T} <: AbstractIntegrator end

step_size(lf::AbstractLeapfrog) = lf.ϵ
jitter(::Union{AbstractRNG,AbstractVector{<:AbstractRNG}}, lf::AbstractLeapfrog) = lf
function temper(
    lf::AbstractLeapfrog, r, ::NamedTuple{(:i, :is_half),<:Tuple{Integer,Bool}}, ::Int
)
    return r
end
stat(lf::AbstractLeapfrog) = (step_size=step_size(lf), nom_step_size=nom_step_size(lf))

update_nom_step_size(lf::AbstractLeapfrog, ϵ) = @set lf.ϵ = ϵ

"""
$(TYPEDEF)

Leapfrog integrator with fixed step size `ϵ`.

# Fields

$(TYPEDFIELDS)
"""
struct Leapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ::T
end
function Base.show(io::IO, mime::MIME"text/plain", l::Leapfrog)
    return print(io, "Leapfrog with step size ϵ=", round.(l.ϵ; sigdigits=3), ")")
end
integrator_eltype(i::AbstractLeapfrog{T}) where {T<:AbstractFloat} = T

### Jittering

"""
$(TYPEDEF)

Leapfrog integrator with randomly "jittered" step size `ϵ` for every trajectory.

# Fields

$(TYPEDFIELDS)

# Description
This is the same as `LeapFrog`(@ref) but with a "jittered" step size. This means 
that at the beginning of each trajectory we sample a step size `ϵ` by adding or 
subtracting from the nominal/base step size `ϵ0` some random proportion of `ϵ0`, 
with the proportion specified by `jitter`, i.e. `ϵ = ϵ0 - jitter * ϵ0 * rand()`.
p
Jittering might help alleviate issues related to poor interactions with a fixed step size:
- In regions with high "curvature" the current choice of step size might mean over-shoot 
  leading to almost all steps being rejected. Randomly sampling the step size at the 
  beginning of the trajectories can therefore increase the probability of escaping such
  high-curvature regions.
- Exact periodicity of the simulated trajectories might occur, i.e. you might be so
  unlucky as to simulate the trajectory forwards in time `L ϵ` and ending up at the
  same point (which results in non-ergodicity; see Section 3.2 in [1]). If momentum
  is refreshed before each trajectory, then this should not happen *exactly* but it
  can still be an issue in practice. Randomly choosing the step-size `ϵ` might help
  alleviate such problems.

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
"""
struct JitteredLeapfrog{FT<:AbstractFloat,T<:AbstractScalarOrVec{FT}} <: AbstractLeapfrog{T}
    "Nominal (non-jittered) step size."
    ϵ0::T
    "The proportion of the nominal step size `ϵ0` that may be added or subtracted."
    jitter::FT
    "Current (jittered) step size."
    ϵ::T
end

JitteredLeapfrog(ϵ0, jitter) = JitteredLeapfrog(ϵ0, jitter, ϵ0)

function Base.show(io::IO, mime::MIME"text/plain", l::JitteredLeapfrog)
    return print(
        io,
        "JitteredLeapfrog with step size ",
        round.(l.ϵ0; sigdigits=3),
        ", jitter ",
        round.(l.jitter; sigdigits=3),
        ", jittered step size ",
        round.(l.ϵ; sigdigits=3),
    )
end

nom_step_size(lf::JitteredLeapfrog) = lf.ϵ0

update_nom_step_size(lf::JitteredLeapfrog, ϵ0) = @set lf.ϵ0 = ϵ0

# Jitter step size; ref: https://github.com/stan-dev/stan/blob/1bb054027b01326e66ec610e95ef9b2a60aa6bec/src/stan/mcmc/hmc/base_hmc.hpp#L177-L178
function jitter(rng::AbstractRNG, lf::JitteredLeapfrog{FT,FT}) where {FT<:AbstractFloat}
    ϵ = lf.ϵ0 * (1 + lf.jitter * (2 * rand(rng, FT) - 1))
    return @set lf.ϵ = ϵ
end
function jitter(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}}, lf::JitteredLeapfrog{FT,T}
) where {FT<:AbstractFloat,T<:AbstractVector{FT}}
    ϵ = similar(lf.ϵ0)
    if rng isa AbstractRNG
        Random.rand!(rng, ϵ)
    else
        @argcheck length(rng) == length(ϵ)
        map!(Base.Fix2(rand, FT), ϵ, rng)
    end
    @. ϵ = lf.ϵ0 * (1 + lf.jitter * (2 * ϵ - 1))
    return @set lf.ϵ = ϵ
end

### Tempering
# TODO: add ref or at least explain what exactly we're doing
"""
$(TYPEDEF)

Tempered leapfrog integrator with fixed step size `ϵ` and "temperature" `α`.

# Fields

$(TYPEDFIELDS)

# Description

Tempering can potentially allow greater exploration of the posterior, e.g. 
in a multi-modal posterior jumps between the modes can be more likely to occur.
"""
struct TemperedLeapfrog{FT<:AbstractFloat,T<:AbstractScalarOrVec{FT}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ::T
    "Temperature parameter."
    α::FT
end

function Base.show(io::IO, mime::MIME"text/plain", l::TemperedLeapfrog)
    return print(
        io,
        "TemperedLeapfrog with step size ϵ=",
        round.(l.ϵ; sigdigits=3),
        " and temperature parameter α=",
        round.(l.α; sigdigits=3),
    )
end

"""
    temper(lf::TemperedLeapfrog, r, step::NamedTuple{(:i, :is_half),<:Tuple{Integer,Bool}}, n_steps::Int)

Tempering step. `step` is a named tuple with
- `i` being the current leapfrog iteration and
- `is_half` indicating whether or not it's (the first) half momentum/tempering step
"""
function temper(
    lf::TemperedLeapfrog,
    r,
    step::NamedTuple{(:i, :is_half),<:Tuple{Integer,Bool}},
    n_steps::Int,
)
    if step.i > n_steps
        throw(BoundsError("Current leapfrog iteration exceeds the total number of steps."))
    end
    i_temper = 2(step.i - 1) + 1 + !step.is_half    # counter for half temper steps
    return i_temper <= n_steps ? r * sqrt(lf.α) : r / sqrt(lf.α)
end

# `step` method for integrators above
# method for `DiffEqIntegrator` is defined in the OrdinaryDiffEq extension
const DefaultLeapfrog{FT<:AbstractFloat,T<:AbstractScalarOrVec{FT}} = Union{
    Leapfrog{T},JitteredLeapfrog{FT,T},TemperedLeapfrog{FT,T}
}
function step(
    lf::DefaultLeapfrog{FT,T},
    h::Hamiltonian,
    z::P,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj}=Val(false),
) where {FT<:AbstractFloat,T<:AbstractScalarOrVec{FT},P<:PhasePoint,FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    ϵ = ϵ'

    if FullTraj
        res = Vector{P}(undef, n_steps)
    end

    (; θ, r) = z
    (; value, gradient) = z.ℓπ
    for i in 1:n_steps
        # Tempering
        r = temper(lf, r, (i=i, is_half=true), n_steps)
        # Take a half leapfrog step for momentum variable
        r = r - ϵ / 2 .* gradient
        # Take a full leapfrog step for position variable
        ∇r = ∂H∂r(h, r)
        θ = θ + ϵ .* ∇r
        # Take a half leapfrog step for momentum variable
        (; value, gradient) = ∂H∂θ(h, θ)
        r = r - ϵ / 2 .* gradient
        # Tempering
        r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ, r; ℓπ=DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                resize!(res, i)
            end
            break
        end
    end
    return if FullTraj
        res
    else
        z
    end
end
