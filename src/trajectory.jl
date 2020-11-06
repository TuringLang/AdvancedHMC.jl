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
struct SliceTS{FT<:AbstractFloat,T<:AbstractScalarOrVec{FT}} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Slice variable in log-space."
    ℓu      ::  T
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
struct MultinomialTS{FT<:AbstractFloat,T<:AbstractScalarOrVec{FT}} <: AbstractTrajectorySampler
    "Sampled candidate `PhasePoint`."
    zcand   ::  PhasePoint
    "Total energy for the given tree, i.e. the sum of energies of all leaves."
    ℓw      ::  T
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
    SliceTS(s::SliceTS, H0::AbstractScalarOrVec{<:AbstractFloat}, zcand::PhasePoint)

Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
function SliceTS(s::SliceTS, H0::AbstractScalarOrVec{<:AbstractFloat}, zcand::PhasePoint)
    return SliceTS(zcand, s.ℓu, ifelse.(s.ℓu .<= .-energy(zcand), 1, 0))
end

"""
    MultinomialTS(s::MultinomialTS, H0::AbstractScalarOrVec{<:AbstractFloat}, zcand::PhasePoint)

Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
function MultinomialTS(s::MultinomialTS, H0::AbstractScalarOrVec{<:AbstractFloat}, zcand::PhasePoint)
    return MultinomialTS(zcand, H0 .- energy(zcand))
end

function combine(rng::AbstractRNG, s1::SliceTS{FT,FT}, s2::SliceTS{FT,FT}) where {FT<:AbstractFloat}
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = rand(rng) < s1.n / n ? s1.zcand : s2.zcand
    return SliceTS(zcand, s1.ℓu, n)
end
function combine(rng::AbstractRNG, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    # ensure is_accept size matches s1.ℓu
    α = s1.n / n
    is_accept = broadcast(s1.ℓu) do _
        rand(rng) < α
    end
    zcand = deepcopy(s1.zcand)
    zcand = accept!(s2.zcand, zcand, is_accept)
    return SliceTS(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::SliceTS, s2::SliceTS)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    return SliceTS(zcand, s1.ℓu, n)
end

function accept!(
    s::SliceTS,
    s′::SliceTS,
    is_accept::AbstractVector{Bool}
)
    zcand = accept!(s.zcand, s′.zcand, is_accept)
    ℓu = accept!(s.ℓu, s′.ℓu, is_accept)
    n = accept!(s.n, s′.n, is_accept)
    return SliceTS(zcand, ℓu, n)
end

function combine(rng::AbstractRNG, s1::MultinomialTS{FT,FT}, s2::MultinomialTS{FT,FT}) where {FT<:AbstractFloat}
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = rand(rng) < exp(s1.ℓw - ℓw) ? s1.zcand : s2.zcand
    return MultinomialTS(zcand, ℓw)
end
function combine(rng::AbstractRNG, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp.(s1.ℓw, s2.ℓw)
    is_accept = rand.(rng) .< exp.(s1.ℓw .- ℓw)
    zcand = deepcopy(s1.zcand)
    zcand = accept!(s2.zcand, zcand, is_accept)
    return MultinomialTS(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::MultinomialTS, s2::MultinomialTS)
    ℓw = logaddexp.(s1.ℓw, s2.ℓw)
    return MultinomialTS(zcand, ℓw)
end

function accept!(
    s::MultinomialTS,
    s′::MultinomialTS,
    is_accept::AbstractVector{Bool}
)
    zcand = accept!(s.zcand, s′.zcand, is_accept)
    ℓw = accept!(s.ℓw, s′.ℓw, is_accept)
    return MultinomialTS(zcand, ℓw)
end

mh_accept(rng::AbstractRNG, s::SliceTS, s′::SliceTS) = rand.(rng) .< min(1, s′.n / s.n)

function mh_accept(rng::AbstractRNG, s::MultinomialTS, s′::MultinomialTS)
    return rand.(rng) .< min.(1, exp.(s′.ℓw .- s.ℓw))
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
    z = accept!(z, z′, is_accept)    # NOTE: this function changes `z′` in place in matrix-parallel mode
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
struct GeneralisedNoUTurn{T<:AbstractVecOrMat{<:Real}} <: AbstractTerminationCriterion
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
struct StrictGeneralisedNoUTurn{T<:AbstractVecOrMat{<:Real}} <: AbstractTerminationCriterion
    "Integral or sum of momenta along the integration path."
    rho::T
end

StrictGeneralisedNoUTurn(z::PhasePoint) = StrictGeneralisedNoUTurn(z.r)

combine(::ClassicNoUTurn, ::ClassicNoUTurn) = ClassicNoUTurn()
combine(cleft::T, cright::T) where {T<:GeneralisedNoUTurn} = T(cleft.rho + cright.rho)
combine(cleft::T, cright::T) where {T<:StrictGeneralisedNoUTurn} = T(cleft.rho + cright.rho)

@inline accept!(
    ::ClassicNoUTurn,
    ::ClassicNoUTurn,
    is_accept::AbstractVector{Bool},
) = ClassicNoUTurn()

@inline function accept!(
    c::CT,
    c′::CT,
    is_accept::AbstractVector{Bool}
) where {T<:AbstractMatrix{<:Real},CT<:Union{GeneralisedNoUTurn{T},StrictGeneralisedNoUTurn{T}}}
    accept!(c.rho, c′.rho, is_accept)
    return c′
end

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
struct Termination{T<:Union{Bool,AbstractVector{Bool}}}
    dynamic::T
    numerical::T
end

Base.show(io::IO, d::Termination) = print(io, "Termination(dynamic=$(d.dynamic), numerical=$(d.numerical))")
Base.:*(d1::Termination, d2::Termination) = Termination(d1.dynamic .| d2.dynamic, d1.numerical .| d2.numerical)
isterminated(d::Termination) = d.dynamic .| d.numerical

"""
    Termination(s::SliceTS, nt::NUTS, H0, H′)

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::SliceTS, nt::NUTS, H0, H′)
    numerical = .!(s.ℓu .< nt.Δ_max .- H′)
    return Termination(zero(numerical), numerical)
end

"""
    Termination(s::MultinomialTS, nt::NUTS, H0, H′)

Check termination of a Hamiltonian trajectory.
"""
function Termination(s::MultinomialTS, nt::NUTS, H0, H′)
    numerical = .!(.-H0 .< nt.Δ_max .- H′)
    return Termination(zero(numerical), numerical)
end

@inline function accept!(
    termination::TT,
    termination′::TT,
    is_accept::AbstractVector{Bool}
) where {T<:AbstractVector{Bool},TT<:Termination{T}}
    accept!(termination.dynamic, termination′.dynamic, is_accept)
    accept!(termination.numerical, termination′.numerical, is_accept)
    return termination′
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
        maxabs.(treeleft.ΔH_max, treeright.ΔH_max),
    )
end

function accept!(
    tree::BinaryTree,
    tree′::BinaryTree,
    is_accept::AbstractVector{Bool}
)
    if any(!, is_accept)
        zleft = accept!(tree.zleft, tree′.zleft, is_accept)
        zright = accept!(tree.zright, tree′.zright, is_accept)
        c = accept!(tree.c, tree′.c, is_accept)
        sum_α = accept!(tree.sum_α, tree′.sum_α, is_accept)
        nα = accept!(tree.nα, tree′.nα, is_accept)
        ΔH_max = accept!(tree.ΔH_max, tree′.ΔH_max, is_accept)
        tree′ = BinaryTree(zleft, zright, c, sum_α, nα, ΔH_max)
    end
    return tree′
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
    s = (colwise_dot(Δθ, ∂H∂r(h, -z0.r)) .>= 0) .| (colwise_dot(-Δθ, ∂H∂r(h, z1.r)) .>= 0)
    return Termination(s, zero(s))
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
    return Termination(s, zero(s))
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
    return Termination(s, zero(s))
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
    return Termination(s, zero(s))
end

function generalised_uturn_criterion(rho, p_sharp_minus, p_sharp_plus)
    return (colwise_dot(rho, p_sharp_minus) .<= 0) .| (colwise_dot(rho, p_sharp_plus) .<= 0)
end

function isterminated(h::Hamiltonian, t::BinaryTree{T}, args...) where {T<:Union{ClassicNoUTurn, GeneralisedNoUTurn}}
    return isterminated(h, t)
end

struct TreeState{T,S,TC}
    tree::T
    sampler::S
    termination::TC
end

@inline isterminated(state::TreeState) = isterminated(state.termination)

function accept!(
    state::TreeState,
    state′::TreeState,
    is_accept::AbstractVector{Bool}
)
    if any(!, is_accept)
        tree′ = accept!(state.tree, state′.tree, is_accept)
        sampler′ = accept!(state.sampler, state′.sampler, is_accept)
        termination′ = accept!(state.termination, state′.termination, is_accept)
        state′ = TreeState(tree′, sampler′, termination′)
    end
    return state′
end

"""
Combine `state` and `state′` for all chains where `state` has not terminated.
"""
function combine(rng::AbstractRNG, h::Hamiltonian, state::TreeState, state′::TreeState, v)
    treeleft, treeright = left_right_subtrees(state.tree, state′.tree, v)
    tree′ = combine(treeleft, treeright)
    sampler′ = combine(rng, state.sampler, state′.sampler)
    termination′ = state.termination * state′.termination * isterminated(h, tree′, treeleft, treeright)
    state′ = TreeState(tree′, sampler′, termination′)
    is_accept = .!isterminated(state)
    state′ = accept!(state, state′, is_accept)
    return state′
end

"""
Combine `state` and `state′` with candidate point `zcand` for all chains where `state` has
not terminated.
"""
function combine(zcand::PhasePoint, h::Hamiltonian, state::TreeState, state′::TreeState, v)
    treeleft, treeright = left_right_subtrees(state.tree, state′.tree, v)
    tree′ = combine(treeleft, treeright)
    sampler′ = combine(zcand, state.sampler, state′.sampler)
    termination′ = state.termination * state′.termination * isterminated(h, tree′, treeleft, treeright)
    state′ = TreeState(tree′, sampler′, termination′)
    is_accept = .!isterminated(state)
    state′ = accept!(state, state′, is_accept)
    return state′
end

"""
Create a level 0 `TreeState` at the initial point `z0`.
"""
function init_tree_state(rng::AbstractRNG, τ::NUTS{S,C}, z0::PhasePoint) where {S,C}
    H0 = energy(z0)
    tree = BinaryTree(z0, z0, C(z0), zero(H0), map(_ -> 0, H0), zero(H0))
    sampler = S(rng, z0)
    # to get same type of array as boolean check
    term = H0 .> Inf
    termination = Termination(term, deepcopy(term))
    state = TreeState(tree, sampler, termination)
    return state
end

function build_one_leaf_tree(nt::NUTS{S,C}, h, z, sampler, v, H0) where {S,C}
    z′ = step(nt.integrator, h, z, 1; fwd = v .> 0)
    H′ = energy(z′)
    ΔH = H′ - H0
    α′ = exp.(min.(0, .-ΔH))
    tree′ = BinaryTree(z′, z′, C(z′), α′, map(_ -> 1, v), ΔH)
    sampler′ = S(sampler, H0, z′)
    termination′ = Termination(sampler′, nt, H0, H′)
    return z′, TreeState(tree′, sampler′, termination′)
end

"""
    subtree_cache_combine_range(ileaf::Integer) -> UnitRange

Get the indices of the cached trees that need to be combined with the two-leaf tree produced
at even leaf number `ileaf`.
"""
@inline function subtree_cache_combine_range(ileaf)
    imin = count_ones(ileaf)
    inum = trailing_zeros(ileaf)
    return (imin + inum - 1):-1:imin
end

function tree_extension_phasepoint(tree, v::Int)
    return v == -1 ? tree.zleft : tree.zright
end
function tree_extension_phasepoint(tree, v)
    isright = isone.(v)
    z′ = deepcopy(tree.zright)
    z′ = accept!(tree.zleft, z′, isright)
    return z′
end

function left_right_subtrees(tree, tree′, v::Int)
    return v == -1 ? (tree′, tree) : (tree, tree′)
end
function left_right_subtrees(tree, tree′, v)
    tree_left, tree_right = deepcopy(tree), deepcopy(tree′)
    is_accept = isone.(v)
    tree_right = accept!(tree, tree_right, is_accept)
    tree_left = accept!(tree′, tree_left, is_accept)
    return tree_left, tree_right
end

"""
Recursivly build a tree for a given depth `j`.
"""
function build_tree(
    rng::AbstractRNG,
    nt::NUTS,
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractTrajectorySampler,
    v::Int,
    j::Int,
    H0::AbstractFloat,
)
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        _, state′ = build_one_leaf_tree(nt, h, z, sampler, v, H0)
        return state′
    else
        # Recursion - build the left and right subtrees.
        state′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !isterminated(state′)
            zextend = tree_extension_phasepoint(state′.tree, v)
            state′′ = build_tree(rng, nt, h, zextend, sampler, v, j - 1, H0)
            state′ = combine(rng, h, state′, state′′, v)
        end
        return state′
    end
end

"""
Iteratively build a tree for a given depth `j`.
"""
function build_tree(
    rng::AbstractRNG,
    nt::NUTS,
    h::Hamiltonian,
    z::PhasePoint{<:AbstractMatrix{<:AbstractFloat}},
    sampler::AbstractTrajectorySampler,
    v::AbstractVector{Int},
    j::Int,
    H0::AbstractVector{<:AbstractFloat},
)
    ileaf_max = 2^j
    state_cache_size = j

    z′, state′ = build_one_leaf_tree(nt, h, z, sampler, v, H0)
    j == 0 && return state′

    has_terminated = isterminated(state′)

    # TODO: allocate state_cache in `transition` and reuse for all subtrees
    state_cache = Vector{typeof(state′)}(undef, state_cache_size)
    icache = 1
    ileaf = 1
    while ileaf < ileaf_max && any(!, has_terminated)
        # cache previous state for future checks
        @inbounds state_cache[icache] = state′

        # take a single step
        z′, state′ = build_one_leaf_tree(nt, h, z′, state′.sampler, v, H0)
        ileaf += 1

        # combine with cached subtrees with same number of leaves
        combine_range = subtree_cache_combine_range(ileaf)
        for i in combine_range
            @inbounds state′ = combine(rng, h, state_cache[i], state′, v)
        end
        icache = last(combine_range)
        has_terminated .|= isterminated(state′)
    end

    # combine even if not same number of leaves
    # this only executes if the above loop was terminated prematurely
    for i in (icache - 1):-1:1
        @inbounds state′ = combine(rng, h, state_cache[i], state′, v)
    end

    return state′
end

function transition(
    rng::AbstractRNG,
    τ::NUTS,
    h::Hamiltonian,
    z0::PhasePoint,
)
    state = init_tree_state(rng, τ, z0)
    H0 = energy(z0)
    zcand = z0

    integrator = jitter(rng, τ.integrator)
    τ = reconstruct(τ, integrator=integrator)

    j = 0
    while !isterminated(state) && j < τ.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        # get point from which next tree is extended
        zextend = tree_extension_phasepoint(state.tree, v)
        # Create a tree with depth `j` from `zextend`
        state′ = build_tree(rng, τ, h, zextend, state.sampler, v, j, H0)

        # Perform a MH step and increse depth if not terminated
        # this check is performed even if unneeded for consistency with iterative NUTS
        is_accept = mh_accept(rng, state.sampler, state′.sampler)
        if !isterminated(state′)
            j = j + 1   # increment tree depth
            if is_accept
                zcand = state′.sampler.zcand
            end
        end

        # Combine the proposed state and the current state (no matter terminated or not)
        state = combine(zcand, h, state, state′, v)
    end

    H = energy(zcand)
    tstat = merge(
        (
            n_steps=state.tree.nα,
            is_accept=true,
            acceptance_rate=state.tree.sum_α / state.tree.nα,
            log_density=zcand.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
            max_hamiltonian_energy_error=state.tree.ΔH_max,
            tree_depth=j,
            numerical_error=state.termination.numerical,
        ),
        stat(τ.integrator),
    )

    return Transition(zcand, tstat)
end

function transition(
    rng::AbstractRNG,
    τ::NUTS,
    h::Hamiltonian,
    z0::PhasePoint{<:AbstractMatrix{<:AbstractFloat}},
)
    state = init_tree_state(rng, τ, z0)
    H0 = energy(z0)
    zcand = deepcopy(z0)

    integrator = jitter(rng, τ.integrator)
    τ = reconstruct(τ, integrator=integrator)

    j = 0
    tree_depth = map(_ -> j, H0)
    has_terminated = isterminated(state)
    while j < τ.max_depth && any(!, has_terminated)
        # Sample a direction; `-1` means left and `1` means right
        v = map(_ -> rand(rng, [-1, 1]), H0)
        # get point from which next tree is extended
        zextend = tree_extension_phasepoint(state.tree, v)
        # Create a tree with depth `j` from `zextend`
        state′ = build_tree(rng, τ, h, zextend, state.sampler, v, j, H0)
        j = j + 1   # increment tree depth

        has_terminated = or!(has_terminated, isterminated(state′))
        # increase tree depth if still hasn't yet terminated
        tree_depth .+= .!has_terminated
        # never accept a proposal from a subtree that has already terminated
        is_reject = has_terminated .| .!mh_accept(rng, state.sampler, state′.sampler)
        zcand′ = state′.sampler.zcand
        zcand = accept!(zcand′, zcand, is_reject)

        state = combine(zcand, h, state, state′, v)
        has_terminated = or!(has_terminated, isterminated(state))
    end

    H = energy(zcand)
    tstat = merge(
        (
            n_steps=state.tree.nα,
            is_accept=true,
            acceptance_rate=state.tree.sum_α ./ state.tree.nα,
            log_density=zcand.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
            max_hamiltonian_energy_error=state.tree.ΔH_max,
            tree_depth=tree_depth,
            numerical_error=state.termination.numerical,
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
