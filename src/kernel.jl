# Developers' Notes
#
# Kai Xu on 6th July 2019:
# Not all functions that use `rng` require a fallback function with `GLOBAL_RNG` as default. 
# In short, only those exported to other libries need such a fallback function. 
# Internal uses shall always use the explict `rng` version.

#######################################
# Container to store transition state #
#######################################

"""
$(TYPEDEF)

A transition that contains the phase point and other statistics of the transition.

## Fields

$(TYPEDFIELDS)
"""
struct Transition{P<:PhasePoint, NT<:NamedTuple}
    "Phase-point for the transition."
    z       ::  P
    "Statistics related to the transition, e.g. energy, acceptance rate, etc."
    stat    ::  NT
end

"Return the statistics for transition `t`."
stat(t::Transition) = t.stat

################################
# Momentum refreshment methods #
################################

abstract type AbstractRefreshment end

"Completly resample new momentum."
struct FullRefreshment <: AbstractRefreshment end

refresh(rng, z, h, ::FullRefreshment) = phasepoint(h, z.θ, rand(rng, h.metric))

"""
Partial momentum refreshment with refresh rate `α`.

## References

1. Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov chain monte carlo 2.11 (2011): 2.
"""
struct PartialRefreshment{F<:AbstractFloat} <: AbstractRefreshment
    α::F
end

refresh(rng, z, h, ref::PartialRefreshment) = 
    phasepoint(h, z.θ, ref.α * z.r + (1 - ref.α^2) * rand(rng, h.metric))

####################################
# Trajectory termination criterion #
####################################

abstract type AbstractTerminationCriterion end
abstract type StaticTerminationCriterion <: AbstractTerminationCriterion end
abstract type DynamicTerminationCriterion <: AbstractTerminationCriterion end

struct FixedNSteps{T<:Int} <: StaticTerminationCriterion
    n_steps::T
end

struct FixedLength{F<:AbstractFloat} <: StaticTerminationCriterion
    λ::F
end

"""
$(TYPEDEF)

Classic No-U-Turn criterion as described in Eq. (9) in [1].

Informally, 
this will terminate the trajectory expansion if continuing the simulation either forwards or
backwards in time will decrease the distance between the left-most and right-most positions.

## Fields

$(TYPEDFIELDS)

## References

1. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
"""
Base.@kwdef struct ClassicNoUTurn{F<:Real} <: DynamicTerminationCriterion 
    max_depth::Int=10
    Δ_max::F=1000
end

"""
$(TYPEDEF)

Generalised No-U-Turn criterion as described in Section A.4.2 in [1].

## Fields

$(TYPEDFIELDS)

## References

1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
"""
Base.@kwdef struct NoUTurn{F<:Real} <: DynamicTerminationCriterion 
    max_depth::Int=10
    Δ_max::F=1000
end

"""
$(TYPEDEF)

Generalised No-U-Turn criterion as described in Section A.4.2 in [1] with 
added U-turn check as described in [2].

## Fields

$(TYPEDFIELDS)

## References

1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
2. [https://github.com/stan-dev/stan/pull/2800](https://github.com/stan-dev/stan/pull/2800)
"""
Base.@kwdef struct StrictNoUTurn{F<:Real} <: DynamicTerminationCriterion 
    max_depth::Int=10
    Δ_max::F=1000
end

######################
# Trajectory sampler #
######################

"Control how to sample a phase point from a simulated trajectory."
abstract type AbstractTrajectorySampler end

"""
$(TYPEDEF)

Sample the end-point of the trajectory with Metropolis-Hasting correction.

## Fields

$(TYPEDFIELDS)
"""
struct MetropolisTS <: AbstractTrajectorySampler end


"""
$(TYPEDEF)

Trajectory slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.

## Fields

$(TYPEDFIELDS)
"""
struct SliceTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Slice variable in log-space."
    ℓu      ::  F
    "Number of acceptable candidates, i.e. those with probability larger than slice variable."
    n       ::  Int
end

Base.show(io::IO, s::SliceTS) = print(io, "SliceTS(ℓu=$(s.ℓu), n=$(s.n))")

"""
$(TYPEDEF)

Multinomial trajectory sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.

## Fields

$(TYPEDFIELDS)
"""
struct MultinomialTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Total energy for the given tree, i.e. the sum of energies of all leaves."
    ℓw      ::  F
end

Base.show(io::IO, s::MultinomialTS) = print(io, "MultinomialTS(ℓw=$(s.ℓw))")

################################################
# Numerically simulated Hamiltonian trajectory #
################################################

"""
$(TYPEDEF)

Hamiltonian trajectory that is simulated numerically until termination.

## Fields

$(TYPEDFIELDS)

## References

1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
"""
struct Trajectory{I<:AbstractIntegrator, TC<:AbstractTerminationCriterion}
    "Integrator used to simulate trajectory."
    integrator::I
    "Criterion to terminate the simulation."
    term_criterion::TC
end

###############
# MCMC Kernel #
###############

abstract type AbstractKernel end

struct HMCKernel{
    R<:AbstractRefreshment, T<:Trajectory, TS<:AbstractTrajectorySampler
} <: AbstractKernel
    refreshment::R
    trajectory::T
    TS::Type{TS}
end

function transition(rng, h, κ::HMCKernel, z)
    @unpack refreshment, trajectory, TS = κ
    integrator = jitter(rng, trajectory.integrator)
    z = refresh(rng, z, h, refreshment)
    return transition(rng, h, Trajectory(integrator, trajectory.term_criterion), TS, z)
end

struct MixtureKernel{
    F<:AbstractFloat, K1<:AbstractKernel, K2<:AbstractKernel
} <: AbstractKernel
    γ::F
    τ1::K1
    τ2::K2
end

function transition(rng, h, κ::MixtureKernel, z)
    @unpack γ, τ1 , τ2 = κ
    τ = rand(rng) < γ ? τ1 : τ2
    return transition(rng, h, τ, z)
end

include("static_trajectory.jl")

include("dynamic_trajectory.jl")

###
### Initialisation of step size
###

"""
A single Hamiltonian integration step.

NOTE: this function is intended to be used in `find_good_stepsize` only.
"""
function A(h, z, ϵ)
    z′ = step(Leapfrog(ϵ), h, z)
    H′ = energy(z′)
    return z′, H′
end

"""
Find a good initial leap-frog step-size via heuristic search.
"""
function find_good_stepsize(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100,
) where {T<:Real}
    # Initialize searching parameters
    ϵ′ = ϵ = T(0.1)
    a_min, a_cross, a_max = T(0.25), T(0.5), T(0.75) # minimal, crossing, maximal accept ratio
    d = T(2.0)
    # Create starting phase point
    r = rand(rng, h.metric) # sample momentum variable
    z = phasepoint(h, θ, r)
    H = energy(z)

    # Make a proposal phase point to decide direction
    z′, H′ = A(h, z, ϵ)
    ΔH = H - H′ # compute the energy difference; `exp(ΔH)` is the MH accept ratio
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        # `direction` being  `1` means MH ratio too high
        #     - this means our step size is too small, thus we increase
        # `direction` being `-1` means MH ratio too small
        #     - this means our step szie is too large, thus we decrease
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        z′, H′ = A(h, z, ϵ)
        ΔH = H - H′
        DEBUG && @debug "Crossing step" direction H′ ϵ "α = $(min(1, exp(ΔH)))"
        if (direction == 1) && !(ΔH > log(a_cross))
            break
        elseif (direction == -1) && !(ΔH < log(a_cross))
            break
        else
            ϵ = ϵ′
        end
    end
    # Note after the for loop,
    # `ϵ` and `ϵ′` are the two neighbour step sizes across `a_cross`.

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method

    ϵ, ϵ′ = ϵ < ϵ′ ? (ϵ, ϵ′) : (ϵ′, ϵ)  # ensure ϵ < ϵ′;
    # Here we want to use a value between these two given the
    # criteria that this value also gives us a MH ratio between `a_min` and `a_max`.
    # This condition is quite mild and only intended to avoid cases where
    # the middle value of `ϵ` and `ϵ′` is too extreme.
    for _ = 1:max_n_iters
        ϵ_mid = middle(ϵ, ϵ′)
        z′, H′ = A(h, z, ϵ_mid)
        ΔH = H - H′
        DEBUG && @debug "Bisection step" H′ ϵ_mid "α = $(min(1, exp(ΔH)))"
        if (exp(ΔH) > a_max)
            ϵ = ϵ_mid
        elseif (exp(ΔH) < a_min)
            ϵ′ = ϵ_mid
        else
            ϵ = ϵ_mid
            break
        end
    end

    return ϵ
end

function find_good_stepsize(
    h::Hamiltonian,
    θ::AbstractVector{<:AbstractFloat};
    max_n_iters::Int=100,
)
    return find_good_stepsize(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)
end
