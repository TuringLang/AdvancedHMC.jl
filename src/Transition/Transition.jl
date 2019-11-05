####
#### Implementation for Hamiltonian dynamics trajectories
####
#### Developers' Notes
####
#### Not all functions that use `rng` requires a callback function with `GLOBAL_RNG`
#### as default. In short, only those exported to other libries need such a callback
#### function. Internal uses shall always use the explict `rng` version. (Kai Xu 6/Jul/19)

"""
A transition that contains the phase point and
other statistics of the transition.
"""
struct Transition{P<:PhasePoint, NT<:NamedTuple}
    z       ::  P
    stat    ::  NT
end

"""
Abstract Markov chain Monte Carlo proposal.
"""
abstract type AbstractProposal end

"""
Hamiltonian dynamics numerical simulation trajectories.
"""
abstract type AbstractTrajectory{I<:AbstractIntegrator} <: AbstractProposal end

##
## Sampling methods for trajectories.
##

"""
Sampler carried during the building of the tree.
"""
abstract type AbstractTrajectorySampler end

"""
Slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.
"""
struct SliceTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    zcand   ::  PhasePoint
    ℓu      ::  F     # slice variable in log space
    n       ::  Int   # number of acceptable candicates, i.e. those with prob larger than slice variable u
end

Base.show(io::IO, s::SliceTS) = print(io, "SliceTS(ℓu=$(s.ℓu), n=$(s.n))")

"""
Multinomial sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.
"""
struct MultinomialTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    zcand   ::  PhasePoint
    ℓw      ::  F     # total energy for the given tree, i.e. sum of energy of all leaves
end

"""
Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
SliceTS(rng::AbstractRNG, z0::PhasePoint) = SliceTS(z0, log(rand(rng)) - energy(z0), 1)

"""
Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
MultinomialTS(rng::AbstractRNG, z0::PhasePoint) = MultinomialTS(z0, zero(energy(z0)))

"""
Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
SliceTS(s::SliceTS, H0::AbstractFloat, zcand::PhasePoint) =
    SliceTS(zcand, s.ℓu, (s.ℓu <= -energy(zcand)) ? 1 : 0)

"""
Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
MultinomialTS(s::MultinomialTS, H0::AbstractFloat, zcand::PhasePoint) =
    MultinomialTS(zcand, H0 - energy(zcand))

function combine(rng::AbstractRNG, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = rand(rng) < s1.n / n ? s1.zcand : s2.zcand
    SliceTS(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    SliceTS(zcand, s1.ℓu, n)
end

function combine(rng::AbstractRNG, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = rand(rng) < exp(s1.ℓw - ℓw) ? s1.zcand : s2.zcand
    MultinomialTS(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    MultinomialTS(zcand, ℓw)
end

mh_accept(
    rng::AbstractRNG,
    s::SliceTS,
    s′::SliceTS
) = rand(rng) < min(1, s′.n / s.n)

mh_accept(
    rng::AbstractRNG,
    s::MultinomialTS,
    s′::MultinomialTS
) = rand(rng) < min(1, exp(s′.ℓw - s.ℓw))

"""
    transition(τ::AbstractTrajectory{I}, h::Hamiltonian, z::PhasePoint)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit callback function for `transition(GLOBAL_RNG, τ, h, z)`
"""
transition(
    τ::AbstractTrajectory{I},
    h::Hamiltonian,
    z::PhasePoint
) where {I<:AbstractIntegrator} = transition(GLOBAL_RNG, τ, h, z)

###
### Actual trajecory implementations
###
include("static.jl")
include("dynamic.jl")

###
### Initialisation of step size
###

"""
Find a good initial leap-frog step-size via heuristic search.
"""
function find_good_eps(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100
) where {T<:Real}
    # Helper function for a single Hamiltonian integration step
    function A(rng, h, z, ϵ)
        z′ = step(rng, Leapfrog(ϵ), h, z)
        H′ = energy(z′)
        return z′, H′
    end
    # Initialize searching parameters
    ϵ′ = ϵ = T(0.1)
    a_min, a_cross, a_max = T(0.25), T(0.5), T(0.75) # minimal, crossing, maximal accept ratio
    d = T(2.0)
    # Create starting phase point
    r = rand(rng, h.metric) # sample momentum variable
    z = phasepoint(h, θ, r)
    H = energy(z)

    # Make a proposal phase point to decide direction
    z′, H′ = A(rng, h, z, ϵ)
    ΔH = H - H′ # compute the energy difference; `exp(ΔH)` is the MH accept ratio
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        # `direction` being  `1` means MH ratio too high
        #     - this means our step size is too small, thus we increase
        # `direction` being `-1` means MH ratio too small
        #     - this means our step szie is too large, thus we decrease
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        z′, H′ = A(rng, h, z, ϵ)
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
        z′, H′ = A(rng, h, z, ϵ_mid)
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

find_good_eps(
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100
) where {T<:Real} = find_good_eps(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)

"""
Perform MH acceptance based on energy, i.e. negative log probability.
"""
function mh_accept_ratio(
    rng::AbstractRNG,
    Horiginal::T,
    Hproposal::T
) where {T<:Real}
    α = min(1.0, exp(Horiginal - Hproposal))
    accept = rand(rng) < α
    return accept, α
end

####
#### Adaption
####

function update(
    h::Hamiltonian,
    τ::AbstractProposal,
    pc::Adaptation.AbstractPreconditioner
)
    metric = renew(h.metric, getM⁻¹(pc))
    h = reconstruct(h, metric=metric)
    return h, τ
end

function update(
    h::Hamiltonian,
    τ::AbstractProposal,
    da::NesterovDualAveraging
)
    integrator = reconstruct(τ.integrator, ϵ=getϵ(da))
    τ = reconstruct(τ, integrator=integrator)
    return h, τ
end

function update(
    h::Hamiltonian,
    τ::AbstractProposal,
    ca::Union{Adaptation.NaiveHMCAdaptor, Adaptation.StanHMCAdaptor}
)
    metric = renew(h.metric, getM⁻¹(ca.pc))
    h = reconstruct(h, metric=metric)
    integrator = reconstruct(τ.integrator, ϵ=getϵ(ca.ssa))
    τ = reconstruct(τ, integrator=integrator)
    return h, τ
end

function update(
    h::Hamiltonian,
    θ::AbstractVector{T}
) where {T<:Real}
    metric = h.metric
    if length(metric) != length(θ)
        metric = typeof(metric)(length(θ))
        h = reconstruct(h, metric=metric)
    end
    return h
end
