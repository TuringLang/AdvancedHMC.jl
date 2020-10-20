####
#### Implementation for Hamiltonian dynamics trajectories
####
#### Developers' Notes
####
#### Not all functions that use `rng` require a fallback function with `GLOBAL_RNG`
#### as default. In short, only those exported to other libries need such a fallback
#### function. Internal uses shall always use the explict `rng` version. (Kai Xu 6/Jul/19)

"""
$(TYPEDEF)

A transition that contains the phase point and
other statistics of the transition.

# Fields

$(TYPEDFIELDS)
"""
struct Transition{P<:PhasePoint, NT<:NamedTuple}
    "Phase-point for the transition."
    z       ::  P
    "Statistics related to the transition, e.g. energy."
    stat    ::  NT
end

"Returns the statistics for transition `t`."
stat(t::Transition) = t.stat

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
Defines how to sample a phase-point from the simulated trajectory.
"""
abstract type AbstractTrajectorySampler end

"""
$(TYPEDEF)

Samples the end-point of the trajectory.
"""
struct EndPointTS <: AbstractTrajectorySampler end

"""
$(TYPEDEF)

Trajectory slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.

# Fields

$(TYPEDFIELDS)
"""
struct SliceTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Slice variable in log-space."
    ℓu      ::  F
    "Number of acceptable candidates, i.e. those with probability larger than slice variable `u`."
    n       ::  Int
end

Base.show(io::IO, s::SliceTS) = print(io, "SliceTS(ℓu=$(s.ℓu), n=$(s.n))")

"""
$(TYPEDEF)

Multinomial trajectory sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.

# Fields

$(TYPEDFIELDS)
"""
struct MultinomialTS{F<:AbstractFloat} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Total energy for the given tree, i.e. the sum of energies of all leaves."
    ℓw      ::  F
end

"""
    SliceTS(rng::AbstractRNG, z0::PhasePoint)

Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
SliceTS(rng::AbstractRNG, z0::PhasePoint) = SliceTS(z0, log(rand(rng)) - energy(z0), 1)

"""
    MultinomialTS(rng::AbstractRNG, z0::PhasePoint)

Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
MultinomialTS(rng::AbstractRNG, z0::PhasePoint) = MultinomialTS(z0, zero(energy(z0)))

"""
    SliceTS(s::SliceTS, H0::AbstractFloat, zcand::PhasePoint)

Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
function SliceTS(s::SliceTS, H0::AbstractFloat, zcand::PhasePoint)
    return SliceTS(zcand, s.ℓu, (s.ℓu <= -energy(zcand)) ? 1 : 0)
end

"""
    MultinomialTS(s::MultinomialTS, H0::AbstractFloat, zcand::PhasePoint)

Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
function MultinomialTS(s::MultinomialTS, H0::AbstractFloat, zcand::PhasePoint)
    return MultinomialTS(zcand, H0 - energy(zcand))
end

function combine(rng::AbstractRNG, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = rand(rng) < s1.n / n ? s1.zcand : s2.zcand
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(rng::AbstractRNG, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = rand(rng) < exp(s1.ℓw - ℓw) ? s1.zcand : s2.zcand
    return MultinomialTS(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    return MultinomialTS(zcand, ℓw)
end

mh_accept(rng::AbstractRNG, s::SliceTS, s′::SliceTS) = rand(rng) < min(1, s′.n / s.n)

function mh_accept(rng::AbstractRNG, s::MultinomialTS, s′::MultinomialTS)
    return rand(rng) < min(1, exp(s′.ℓw - s.ℓw))
end

"""
    transition(τ::AbstractTrajectory{I}, h::Hamiltonian, z::PhasePoint)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit fallback function for `transition(GLOBAL_RNG, τ, h, z)`
"""
function transition(τ::AbstractTrajectory, h::Hamiltonian, z::PhasePoint)
    return transition(GLOBAL_RNG, τ, h, z)
end

###
### Actual trajectory implementations
###

"""
$(TYPEDEF)

Static HMC with a fixed number of leapfrog steps.

# Fields

$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901))
"""
struct StaticTrajectory{S<:AbstractTrajectorySampler, I<:AbstractIntegrator} <: AbstractTrajectory{I}
    "Integrator used to simulate trajectory."
    integrator  ::  I
    "Number of steps to simulate, i.e. length of trajectory will be `n_steps + 1`."
    n_steps     ::  Int
end

function Base.show(io::IO, τ::StaticTrajectory{<:EndPointTS})
    print(io, "StaticTrajectory{EndPointTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")
end

function Base.show(io::IO, τ::StaticTrajectory{<:MultinomialTS})
    print(io, "StaticTrajectory{MultinomialTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")
end

StaticTrajectory{S}(integrator::I, n_steps::Int) where {S,I} = StaticTrajectory{S,I}(integrator, n_steps)
StaticTrajectory(args...) = StaticTrajectory{EndPointTS}(args...) # default StaticTrajectory using last point from trajectory

function transition(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    τ::StaticTrajectory,
    h::Hamiltonian,
    z::PhasePoint,
)
    H0 = energy(z)

    integrator = jitter(rng, τ.integrator)
    τ = reconstruct(τ, integrator=integrator)

    z′, is_accept, α = sample_phasepoint(rng, τ, h, z)
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
            hamiltonian_energy_error=H - H0,
        ),
        stat(integrator),
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
    is_reject = (!).(is_accept)
    if any(is_reject)
        z′.θ[:,is_reject] = z.θ[:,is_reject]
        z′.r[:,is_reject] = z.r[:,is_reject]
        z′.ℓπ.value[is_reject] = z.ℓπ.value[is_reject]
        z′.ℓπ.gradient[:,is_reject] = z.ℓπ.gradient[:,is_reject]
        z′.ℓκ.value[is_reject] = z.ℓκ.value[is_reject]
        z′.ℓκ.gradient[:,is_reject] = z.ℓκ.gradient[:,is_reject]
    end
    # Always return `z′` as any unaccepted proposal is already reverted
    # NOTE: This in place treatment of `z′` is for memory efficient consideration.
    #       We can also copy `z′ and avoid mutating the original `z′`. But this is
    #       not efficient and immutability of `z′` is not important in this local scope.
    return z′
end

### Use end-point from the trajectory as a proposal and apply MH correction

function sample_phasepoint(rng, τ::StaticTrajectory{EndPointTS}, h, z)
    z′ = step(τ.integrator, h, z, τ.n_steps)
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    return z′, is_accept, α
end

### Multinomial sampling from trajectory

function randcat(rng::AbstractRNG, zs::AbstractVector{<:PhasePoint}, unnorm_ℓp::AbstractVector)
    p = exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp))
    i = randcat(rng, p)
    return zs[i]
end

# zs is in the form of Vector{PhasePoint{Matrix}} and has shape [n_steps][dim, n_chains]
function randcat(rng, zs::AbstractVector{<:PhasePoint}, unnorm_ℓP::AbstractMatrix)
    z = similar(first(zs))
    P = exp.(unnorm_ℓP .- logsumexp(unnorm_ℓP; dims=2)) # (n_chains, n_steps)
    is = randcat(rng, P')
    foreach(enumerate(is)) do (i_chain, i_step)
        zi = zs[i_step]
        z.θ[:,i_chain] = zi.θ[:,i_chain]
        z.r[:,i_chain] = zi.r[:,i_chain]
        z.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z.ℓπ.gradient[:,i_chain] = zi.ℓπ.gradient[:,i_chain]
        z.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z.ℓκ.gradient[:,i_chain] = zi.ℓκ.gradient[:,i_chain]
    end
    return z
end

function sample_phasepoint(rng, τ::StaticTrajectory{MultinomialTS}, h, z)
    n_steps = abs(τ.n_steps)
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = rand_coupled(rng, 0:n_steps) 
    zs_fwd = step(τ.integrator, h, z, n_steps_fwd; fwd=true, full_trajectory=Val(true))
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = step(τ.integrator, h, z, n_steps_bwd; fwd=false, full_trajectory=Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    z′ = randcat(rng, zs, unnorm_ℓprob)
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

abstract type DynamicTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed total trajectory length.
###

"""
$(TYPEDEF)

Standard HMC implementation with fixed total trajectory length.

# Fields

$(TYPEDFIELDS)

# References
1. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov chain Monte Carlo, 2(11), 2. ([arXiv](https://arxiv.org/pdf/1206.1901)) 
"""
struct HMCDA{S<:AbstractTrajectorySampler,I<:AbstractIntegrator} <: DynamicTrajectory{I}
    "Integrator used to simulate trajectory."
    integrator  ::  I
    "Total length of the trajectory, i.e. take `floor(λ / integrator_step)` number of leapfrog steps."
    λ           ::  AbstractFloat
end

function Base.show(io::IO, τ::HMCDA{<:EndPointTS})
    print(io, "HMCDA{EndPointTS}(integrator=$(τ.integrator), λ=$(τ.λ)))")
end
function Base.show(io::IO, τ::HMCDA{<:MultinomialTS})
    print(io, "HMCDA{MultinomialTS}(integrator=$(τ.integrator), λ=$(τ.λ)))")
end

HMCDA{S}(integrator::I, λ::AbstractFloat) where {S,I} = HMCDA{S,I}(integrator, λ)
HMCDA(args...) = HMCDA{EndPointTS}(args...) # default HMCDA using last point from trajectory

function transition(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    τ::HMCDA{S},
    h::Hamiltonian,
    z::PhasePoint,
) where {S}
    # Create the corresponding static τ
    n_steps = max(1, floor(Int, τ.λ / nom_step_size(τ.integrator)))
    static_τ = StaticTrajectory{S}(τ.integrator, n_steps)
    return transition(rng, static_τ, h, z)
end

###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

##
## Variants of no-U-turn criteria
##

abstract type AbstractTerminationCriterion end

"""
$(TYPEDEF)

Classic No-U-Turn criterion as described in Eq. (9) in [1].

Informally, this will terminate the trajectory expansion if continuing
the simulation either forwards or backwards in time will decrease the
distance between the left-most and right-most positions.

# References
1. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623. ([arXiv](http://arxiv.org/abs/1111.4246))
"""
struct ClassicNoUTurn <: AbstractTerminationCriterion end

ClassicNoUTurn(::PhasePoint) = ClassicNoUTurn()

"""
$(TYPEDEF)

Generalised No-U-Turn criterion as described in Section A.4.2 in [1].

# Fields

$(TYPEDFIELDS)

# References
1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
"""
struct GeneralisedNoUTurn{T<:AbstractVector{<:Real}} <: AbstractTerminationCriterion
    "Integral or sum of momenta along the integration path."
    rho::T
end

GeneralisedNoUTurn(z::PhasePoint) = GeneralisedNoUTurn(z.r)

"""
$(TYPEDEF)

Generalised No-U-Turn criterion as described in Section A.4.2 in [1] with 
added U-turn check as described in [2].

# Fields

$(TYPEDFIELDS)

# References
1. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
2. [https://github.com/stan-dev/stan/pull/2800](https://github.com/stan-dev/stan/pull/2800)
"""
struct StrictGeneralisedNoUTurn{T<:AbstractVector{<:Real}} <: AbstractTerminationCriterion
    "Integral or sum of momenta along the integration path."
    rho::T
end

StrictGeneralisedNoUTurn(z::PhasePoint) = StrictGeneralisedNoUTurn(z.r)

combine(::ClassicNoUTurn, ::ClassicNoUTurn) = ClassicNoUTurn()
combine(cleft::T, cright::T) where {T<:GeneralisedNoUTurn} = T(cleft.rho + cright.rho)
combine(cleft::T, cright::T) where {T<:StrictGeneralisedNoUTurn} = T(cleft.rho + cright.rho)


##
## NUTS
##

"""
Dynamic trajectory HMC using the no-U-turn termination criteria algorithm.
"""
struct NUTS{
    S<:AbstractTrajectorySampler,
    C<:AbstractTerminationCriterion,
    I<:AbstractIntegrator,
    F<:AbstractFloat
} <: DynamicTrajectory{I}
    integrator      ::  I
    max_depth       ::  Int
    Δ_max           ::  F
end

function Base.show(io::IO, τ::NUTS{<:SliceTS, <:ClassicNoUTurn})
    print(io, "NUTS{SliceTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:SliceTS, <:GeneralisedNoUTurn})
    print(io, "NUTS{SliceTS,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:MultinomialTS, <:ClassicNoUTurn})
    print(io, "NUTS{MultinomialTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:MultinomialTS, <:GeneralisedNoUTurn})
    print(io, "NUTS{MultinomialTS,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end


const NUTS_DOCSTR = """
    NUTS{S,C}(
        integrator::I,
        max_depth::Int=10,
        Δ_max::F=1000.0
    ) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}

Create an instance for the No-U-Turn sampling algorithm.
"""

"$NUTS_DOCSTR"
function NUTS{S,C}(
    integrator::I,
    max_depth::Int=10,
    Δ_max::F=1000.0,
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    return NUTS{S,C,I,F}(integrator, max_depth, Δ_max)
end

"""
    NUTS(args...) = NUTS{MultinomialTS,GeneralisedNoUTurn}(args...)

Create an instance for the No-U-Turn sampling algorithm
with multinomial sampling and original no U-turn criterion.

Below is the doc for NUTS{S,C}.

$NUTS_DOCSTR
"""
NUTS(args...) = NUTS{MultinomialTS, GeneralisedNoUTurn}(args...)

###
### The doubling tree algorithm for expanding trajectory.
###

"""
    Termination

Termination reasons
- `dynamic`: due to stoping criteria
- `numerical`: due to large energy deviation from starting (possibly numerical errors)
"""
struct Termination
    dynamic::Bool
    numerical::Bool
end

Base.show(io::IO, d::Termination) = print(io, "Termination(dynamic=$(d.dynamic), numerical=$(d.numerical))")
Base.:*(d1::Termination, d2::Termination) = Termination(d1.dynamic || d2.dynamic, d1.numerical || d2.numerical)
isterminated(d::Termination) = d.dynamic || d.numerical

"""
    Termination(s::SliceTS, nt::NUTS, H0::F, H′::F) where {F<:AbstractFloat}

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::SliceTS, nt::NUTS, H0::F, H′::F) where {F<:AbstractFloat}
    return Termination(false, !(s.ℓu < nt.Δ_max + -H′))
end

"""
    Termination(s::MultinomialTS, nt::NUTS, H0::F, H′::F) where {F<:AbstractFloat}

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::MultinomialTS, nt::NUTS, H0::F, H′::F) where {F<:AbstractFloat}
    return Termination(false, !(-H0 < nt.Δ_max + -H′))
end

"""
A full binary tree trajectory with only necessary leaves and information stored.
"""
struct BinaryTree{C<:AbstractTerminationCriterion}
    zleft   # left most leaf node
    zright  # right most leaf node
    c::C    # termination criterion
    sum_α   # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα      # total # of leap frog steps, i.e. phase points in a trajectory
    ΔH_max  # energy in tree with largest absolute different from initial energy
end

"""
    maxabs(a, b)

Return the value with the largest absolute value.
"""
maxabs(a, b) = abs(a) > abs(b) ? a : b

"""
    combine(treeleft::BinaryTree, treeright::BinaryTree)

Merge a left tree `treeleft` and a right tree `treeright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
function combine(treeleft::BinaryTree, treeright::BinaryTree)
    return BinaryTree(
        treeleft.zleft,
        treeright.zright,
        combine(treeleft.c, treeright.c),
        treeleft.sum_α + treeright.sum_α,
        treeleft.nα + treeright.nα,
        maxabs(treeleft.ΔH_max, treeright.ΔH_max),
    )
end

"""
    isterminated(h::Hamiltonian, t::BinaryTree{<:ClassicNoUTurn})

Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the (original) no-U-turn cirterion.

Ref: https://arxiv.org/abs/1111.4246, https://arxiv.org/abs/1701.02434
"""
function isterminated(h::Hamiltonian, t::BinaryTree{<:ClassicNoUTurn})
    # z0 is starting point and z1 is ending point
    z0, z1 = t.zleft, t.zright
    Δθ = z1.θ - z0.θ
    s = (dot(Δθ, ∂H∂r(h, -z0.r)) >= 0) || (dot(-Δθ, ∂H∂r(h, z1.r)) >= 0)
    return Termination(s, false)
end

"""
    isterminated(h::Hamiltonian, t::BinaryTree{<:GeneralisedNoUTurn})

Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion.

Ref: https://arxiv.org/abs/1701.02434
"""
function isterminated(h::Hamiltonian, t::BinaryTree{<:GeneralisedNoUTurn})
    rho = t.c.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

"""
    isterminated(
        h::Hamiltonian, t::T, tleft::T, tright::T
    ) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}

Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion with additional U-turn checks.

Ref: https://arxiv.org/abs/1701.02434 https://github.com/stan-dev/stan/pull/2800
"""
function isterminated(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}
    # Classic generalised U-turn check
    t_generalised = BinaryTree(
        t.zleft,
        t.zright,
        GeneralisedNoUTurn(t.c.rho),
        t.sum_α,
        t.nα,
        t.ΔH_max
    )
    s1 = isterminated(h, t_generalised)

    # U-turn checks for left and right subtree
    # See https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/33
    # for a visualisation.
    s2 = check_left_subtree(h, t, tleft, tright)
    s3 = check_right_subtree(h, t, tleft, tright)
    return s1 * s2 * s3
end

"""
    check_left_subtree(
        h::Hamiltonian, t::T, tleft::T, tright::T
    ) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}

Do a U-turn check between the leftmost phase point of `t` and the leftmost 
phase point of `tright`, the right subtree.
"""
function check_left_subtree(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}
    rho = tleft.c.rho + tright.zleft.r
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, tright.zleft.r))
    return Termination(s, false)
end

"""
    check_left_subtree(
        h::Hamiltonian, t::T, tleft::T, tright::T
    ) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}

Do a U-turn check between the rightmost phase point of `t` and the rightmost
phase point of `tleft`, the left subtree.
"""
function check_right_subtree(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictGeneralisedNoUTurn}}
    rho = tleft.zright.r + tright.c.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, tleft.zright.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

function generalised_uturn_criterion(rho, p_sharp_minus, p_sharp_plus)
    return (dot(rho, p_sharp_minus) <= 0) || (dot(rho, p_sharp_plus) <= 0)
end

function isterminated(h::Hamiltonian, t::BinaryTree{T}, args...) where {T<:Union{ClassicNoUTurn, GeneralisedNoUTurn}}
    return isterminated(h, t)
end

"""
Recursivly build a tree for a given depth `j`.
"""
function build_tree(
    rng::AbstractRNG,
    nt::NUTS{S,C,I,F},
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractTrajectorySampler,
    v::Int,
    j::Int,
    H0::AbstractFloat,
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = energy(z′)
        ΔH = H′ - H0
        α′ = exp(min(0, -ΔH))
        sampler′ = S(sampler, H0, z′)
        return BinaryTree(z′, z′, C(z′), α′, 1, ΔH), sampler′, Termination(sampler′, nt, H0, H′)
    else
        # Recursion - build the left and right subtrees.
        tree′, sampler′, termination′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !isterminated(termination′)
            # Expand left
            if v == -1
                tree′′, sampler′′, termination′′ = build_tree(rng, nt, h, tree′.zleft, sampler, v, j - 1, H0) # left tree
                treeleft, treeright = tree′′, tree′
            # Expand right
            else
                tree′′, sampler′′, termination′′ = build_tree(rng, nt, h, tree′.zright, sampler, v, j - 1, H0) # right tree
                treeleft, treeright = tree′, tree′′
            end
            tree′ = combine(treeleft, treeright)
            sampler′ = combine(rng, sampler′, sampler′′)
            termination′ = termination′ * termination′′ * isterminated(h, tree′, treeleft, treeright)
        end
        return tree′, sampler′, termination′
    end
end

function transition(
    rng::AbstractRNG,
    τ::NUTS{S,C,I,F},
    h::Hamiltonian,
    z0::PhasePoint,
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    H0 = energy(z0)
    tree = BinaryTree(z0, z0, C(z0), zero(F), zero(Int), zero(H0))
    sampler = S(rng, z0)
    termination = Termination(false, false)
    zcand = z0

    integrator = jitter(rng, τ.integrator)
    τ = reconstruct(τ, integrator=integrator)

    j = 0
    while !isterminated(termination) && j < τ.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            tree′, sampler′, termination′ = build_tree(rng, τ, h, tree.zleft, sampler, v, j, H0)
            treeleft, treeright = tree′, tree
        else
            # Create a tree with depth `j` on the right
            tree′, sampler′, termination′ = build_tree(rng, τ, h, tree.zright, sampler, v, j, H0)
            treeleft, treeright = tree, tree′
        end
        # Perform a MH step and increse depth if not terminated
        if !isterminated(termination′)
            j = j + 1   # increment tree depth
            if mh_accept(rng, sampler, sampler′)
                zcand = sampler′.zcand
            end
        end
        # Combine the proposed tree and the current tree (no matter terminated or not)
        tree = combine(treeleft, treeright)
        # Update sampler
        sampler = combine(zcand, sampler, sampler′)
        # update termination
        termination = termination * termination′ * isterminated(h, tree, treeleft, treeright)
    end

    H = energy(zcand)
    tstat = merge(
        (
            n_steps=tree.nα,
            is_accept=true,
            acceptance_rate=tree.sum_α / tree.nα,
            log_density=zcand.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
            max_hamiltonian_energy_error=tree.ΔH_max,
            tree_depth=j,
            numerical_error=termination.numerical,
        ),
        stat(τ.integrator),
    )

    return Transition(zcand, tstat)
end

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

"""
Perform MH acceptance based on energy, i.e. negative log probability.
"""
function mh_accept_ratio(
    rng::AbstractRNG,
    Horiginal::T,
    Hproposal::T,
) where {T<:AbstractFloat}
    α = min(one(T), exp(Horiginal - Hproposal))
    accept = rand(rng, T) < α
    return accept, α
end

function mh_accept_ratio(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    Horiginal::AbstractVector{<:T},
    Hproposal::AbstractVector{<:T},
) where {T<:AbstractFloat}
    α = min.(one(T), exp.(Horiginal .- Hproposal))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    accept = rand(rng, T, length(Horiginal)) .< α
    return accept, α
end
