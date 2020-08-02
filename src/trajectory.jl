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

struct FixedNSteps{T<:Integer} <: StaticTerminationCriterion
    L::T
end

struct FixedLength{F<:AbstractFloat} <: StaticTerminationCriterion
    t::F
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
@with_kw struct ClassicNoUTurn{F<:Real} <: DynamicTerminationCriterion 
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
@with_kw struct NoUTurn{F<:Real} <: DynamicTerminationCriterion 
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
@with_kw struct StrictNoUTurn{F<:Real} <: DynamicTerminationCriterion 
    max_depth::Int=10
    Δ_max::F=1_000
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
    criterion::TC
end

Base.show(io::IO, τ::Trajectory) =
    print(io, "Trajectory(integrator=$(τ.integrator), criterion=$(τ.criterion))")

###############
# MCMC Kernel #
###############

abstract type AbstractKernel end

struct HMCKernel{
    R<:AbstractRefreshment, T<:Trajectory, TS<:AbstractTrajectorySampler
} <: AbstractKernel
    refreshment::R
    τ::T
    TS::Type{TS}
end

Base.show(io::IO, κ::HMCKernel) =
    print(io, "HMCKernel(\n    refreshment=$(κ.refreshment),\n    τ=$(κ.τ),\n    TS=$(κ.TS)\n)")

function transition(rng, h, κ::HMCKernel, z)
    @unpack refreshment, τ, TS = κ
    τ = reconstruct(τ, integrator=jitter(rng, τ.integrator))
    z = refresh(rng, z, h, refreshment)
    return transition(rng, h, τ, TS, z)
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

##########
# Static #
##########

function transition(rng, h, τ::Trajectory{I, <:FixedNSteps}, ::Type{TS}, z) where {I, TS}
    @unpack integrator, criterion = τ
    H0 = energy(z)
    z′, is_accept, α = propose_phasepoint(rng, integrator, criterion, TS, h, z)
    # Do the actual accept / reject
    # NOTE: this function changes `z′` in-place in the vectorized mode
    z = accept_phasepoint!(z, z′, is_accept)
    # Reverse momentum variable to preserve reversibility
    z = PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    H = energy(z)
    tstat = merge(
        (
            n_steps = criterion.L,
            is_accept = is_accept,
            acceptance_rate = α,
            log_density = z.ℓπ.value,
            hamiltonian_energy = H,
            hamiltonian_energy_error = H - H0,
        ),
        stat(integrator),
    )
    return Transition(z, tstat)
end

function transition(rng, h, τ::Trajectory{I, <:FixedLength}, ::Type{TS}, z) where {I, TS}
    @unpack integrator, criterion = τ
    # Create the corresponding `FixedNSteps` criterion
    L = max(1, floor(Int, criterion.t / nom_step_size(integrator)))
    τ = Trajectory(integrator, FixedNSteps(L))
    return transition(rng, h, τ, TS, z)
end

"Use end-point from the trajectory as a proposal and apply MH correction"
function propose_phasepoint(rng, integrator, tc, ::Type{MetropolisTS}, h, z)
    z′ = step(integrator, h, z, tc.L)
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    return z′, is_accept, α
end

"Perform MH acceptance based on energy, i.e. negative log probability"
function mh_accept_ratio(rng, H::R, H′::R) where {R<:Real}
    α = min(one(R), exp(H - H′))
    accept = rand(rng, R) < α
    return accept, α
end

function mh_accept_ratio(rng, H::V, H′::V) where {R<:Real, V<:AbstractVector{R}}
    α = min.(one(R), exp.(H .- H′))
    # NOTE: There is a chance that sharing the RNG over multiple
    #       chains for accepting / rejecting might couple
    #       the chains. We need to revisit this more rigirously 
    #       in the future. See discussions at 
    #       https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    accept = rand(rng, R, length(H)) .< α
    return accept, α
end

"Propose a point from the trajectory using Multinomial sampling"
function propose_phasepoint(rng, integrator, tc, ::Type{MultinomialTS}, h, z)
    L = abs(tc.L)
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    L_fwd = rand_coupled(rng, 0:L)
    L_bwd = L - L_fwd
    zs_fwd = step(integrator, h, z, L_fwd; fwd=true,  full_trajectory=Val(true))
    zs_bwd = step(integrator, h, z, L_bwd; fwd=false, full_trajectory=Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    z′ = multinomial_sample(rng, zs, unnorm_ℓprob)
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

"Sample `i` from a Categorical with unnormalised probability `unnorm_ℓp` and return `zs[i]`"
function multinomial_sample(rng, zs, unnorm_ℓp::AbstractVector)
    p = exp.(unnorm_ℓp .- logsumexp(unnorm_ℓp))
    i = randcat(rng, p)
    return zs[i]
end

# Note: zs is in the form of Vector{PhasePoint{Matrix}} and has shape [n_steps][dim, n_chains]
function multinomial_sample(rng, zs, unnorm_ℓP::AbstractMatrix)
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

accept_phasepoint!(z::T, z′::T, is_accept) where {T<:PhasePoint{<:AbstractVector}} = 
    is_accept ? z′ : z

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
    # NOTE: This in place treatment of `z′` is for memory efficient consideration.
    #       We can also copy `z′ and avoid mutating the original `z′`. But this is
    #       not efficient and immutability of `z′` is not important in this local scope.
    return z′
end

###########
# Dynamic #
###########

################################
# Trajectory sampler utilities #
################################

"Slice sampler for the starting single leaf tree. Slice variable is initialized."
SliceTS(rng::AbstractRNG, z0::PhasePoint) = SliceTS(z0, log(rand(rng)) - energy(z0), 1)

"""
Create a slice sampler for a single leaf tree:
- The slice variable is copied from the passed-in sampler `s`.
- The number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
function SliceTS{T}(ts::SliceTS, H0::T, zcand::PhasePoint) where {T<:AbstractFloat}
    return SliceTS(zcand, ts.ℓu, (ts.ℓu <= -energy(zcand)) ? 1 : 0)
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

"""
Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

## References 

1. https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
MultinomialTS(rng::AbstractRNG, z0::PhasePoint) = MultinomialTS(z0, zero(energy(z0)))

"""
Multinomial sampler for a trajectory consisting only a leaf node.
- The tree weight is the (unnormalised) energy of the leaf.
"""
function MultinomialTS{T}(::MultinomialTS, H0::T, zcand::PhasePoint) where {T<:AbstractFloat}
    return MultinomialTS(zcand, H0 - energy(zcand))
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

#########################
# Termination utilities #
#########################

"Termination statistics"
struct Termination
    "Is it terminated due to stoping criteria?"
    dynamic::Bool
    "Is it terminated due to large energy deviation from starting (possibly numerical errors)"
    numerical::Bool
end

Base.:*(t1::Termination, t2::Termination) = 
    Termination(t1.dynamic || t2.dynamic, t1.numerical || t2.numerical)

isterminated(t::Termination) = t.dynamic || t.numerical

"Check termination of a Hamiltonian trajectory."
Termination(s::SliceTS, tc, H0, H′) = Termination(false, !(s.ℓu < tc.Δ_max + -H′))
Termination(s::MultinomialTS, tc, H0, H′) = Termination(false, !(-H0 < tc.Δ_max + -H′))

"Check U-turn."
check_uturn(rho, pleft, pright) = (dot(rho, pleft) <= 0) || (dot(rho, pright) <= 0)

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the (original) no-U-turn cirterion.

Ref: https://arxiv.org/abs/1111.4246, https://arxiv.org/abs/1701.02434
"""
function isterminated(::ClassicNoUTurn, h, t)
    z0, z1 = t.zleft, t.zright # z0 is the starting point and z1 is the ending point
    s = check_uturn(z1.θ - z0.θ, ∂H∂r(h, z0.r), ∂H∂r(h, z1.r))
    return Termination(s, false)
end

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion.

Ref: https://arxiv.org/abs/1701.02434
"""
function isterminated(::Union{NoUTurn, StrictNoUTurn}, h, t)
    rho = t.rho
    s = check_uturn(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

"`ClassicNoUTurn` and `NoUTurn` only use the merged tree."
isterminated(tc::Union{ClassicNoUTurn, NoUTurn}, h, t, tleft, tright) = isterminated(tc, h, t)

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion with additional U-turn checks.

## References 

1. https://arxiv.org/abs/1701.02434 
2. https://github.com/stan-dev/stan/pull/2800
3. https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/33
"""
function isterminated(tc::StrictNoUTurn, h, t, tleft, tright, testmode=Val(false))
    # Step 0: original generalised U-turn check
    s = isterminated(tc, h, t)

    # Step 1 & 2: U-turn checks for left and right subtree; see [3] for a visualisation.
    # Step 1: Check U-turn between the leftmost phase point of `t` and 
    #         the leftmost  phase point of `tright`, the right subtree.
    rho_left = tleft.rho + tright.zleft.r
    s_left = Termination(
        check_uturn(rho_left, ∂H∂r(h, t.zleft.r), ∂H∂r(h, tright.zleft.r)), false
    )
    # Step 2: Check U-turn between the rightmost phase point of `t` and
    #         the rightmost phase point of `tleft`, the left subtree.
    rho_right = tleft.zright.r + tright.rho
    s_right = Termination(
        check_uturn(rho_right, ∂H∂r(h, tleft.zright.r), ∂H∂r(h, t.zright.r)), false
    )
    if testmode isa Val{false}
        return s * s_left * s_right
    else
        return (s, s_left, s_right)
    end
end

######################
# No-U-turn samplers #
######################

"A trajectory stored as a binary tree with only necessary leaves and information."
struct BinaryTree{R<:Union{Nothing, AbstractVecOrMat}}
    zleft   # left most leaf node
    zright  # right most leaf node
    rho::R  # termination statistic
    sum_α   # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα      # total # of leap frog steps, i.e. phase points in a trajectory
    ΔH_max  # energy in tree with largest absolute different from initial energy
end

"Initialize termination statistic"
rho_init(::ClassicNoUTurn, z) = nothing
rho_init(::Union{NoUTurn, StrictNoUTurn}, z) = z.r

function BinaryTree(zleft, zright, tc::C, sum_α, nα, ΔH_max) where {C<:Union{ClassicNoUTurn, NoUTurn, StrictNoUTurn}}
    @assert zleft == zright
    return BinaryTree(zleft, zright, rho_init(tc, zleft), sum_α, nα, ΔH_max)
end

"Merge two termination statistics"
rho_merge(::T, ::T) where {T<:Nothing} = nothing
rho_merge(rholeft, rhoright) = rholeft + rhoright

"Return the value with the largest absolute value."
maxabs(a, b) = abs(a) > abs(b) ? a : b

"""
Merge a left tree `treeleft` and a right tree `treeright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
function combine(treeleft::BinaryTree, treeright::BinaryTree)
    return BinaryTree(
        treeleft.zleft,
        treeright.zright,
        rho_merge(treeleft.rho, treeright.rho),
        treeleft.sum_α + treeright.sum_α,
        treeleft.nα + treeright.nα,
        maxabs(treeleft.ΔH_max, treeright.ΔH_max),
    )
end

"Recursivly build a tree for a given depth `j`."
function build_tree(rng, int, tc, h, z, sampler::TS, v, j, H0) where {I, C, TS}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(int, h, z, v)
        H′ = energy(z′)
        ΔH = H′ - H0
        α′ = exp(min(0, -ΔH))
        sampler′ = TS(sampler, H0, z′)
        return BinaryTree(z′, z′, tc, α′, 1, ΔH), sampler′, Termination(sampler′, tc, H0, H′)
    else
        # Recursion - build the left and right subtrees.
        tree′, sampler′, termination′ = build_tree(rng, int, tc, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !isterminated(termination′)
            # Expand left
            if v == -1
                tree′′, sampler′′, termination′′ = build_tree(rng, int, tc, h, tree′.zleft, sampler, v, j - 1, H0) # left tree
                treeleft, treeright = tree′′, tree′
            # Expand right
            else
                tree′′, sampler′′, termination′′ = build_tree(rng, int, tc, h, tree′.zright, sampler, v, j - 1, H0) # right tree
                treeleft, treeright = tree′, tree′′
            end
            tree′ = combine(treeleft, treeright)
            sampler′ = combine(rng, sampler′, sampler′′)
            termination′ = termination′ * termination′′ * isterminated(tc, h, tree′, treeleft, treeright)
        end
        return tree′, sampler′, termination′
    end
end

function transition(
    rng, h, τ::Trajectory{I, C}, ::Type{TS}, z0
) where {I, C<:DynamicTerminationCriterion, TS}
    @unpack integrator, criterion = τ
    H0 = energy(z0)
    tree = BinaryTree(z0, z0, criterion, zero(H0), zero(Int), zero(H0))
    sampler = TS(rng, z0)
    termination = Termination(false, false)
    zcand = z0
    j = 0
    while !isterminated(termination) && j < criterion.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            tree′, sampler′, termination′ = build_tree(rng, integrator, criterion, h, tree.zleft, sampler, v, j, H0)
            treeleft, treeright = tree′, tree
        else
            # Create a tree with depth `j` on the right
            tree′, sampler′, termination′ = build_tree(rng, integrator, criterion, h, tree.zright, sampler, v, j, H0)
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
        termination = termination * termination′ * isterminated(criterion, h, tree, treeleft, treeright)
    end

    H = energy(zcand)
    tstat = merge(
        (
            n_steps = tree.nα,
            is_accept = true,
            acceptance_rate = tree.sum_α / tree.nα,
            log_density = zcand.ℓπ.value,
            hamiltonian_energy = H,
            hamiltonian_energy_error = H - H0,
            max_hamiltonian_energy_error = tree.ΔH_max,
            tree_depth = j,
            numerical_error = termination.numerical,
        ),
        stat(integrator),
    )

    return Transition(zcand, tstat)
end

mh_accept(rng, s::TS, s′::TS) where {TS<:SliceTS} = rand(rng) < min(1, s′.n / s.n)
mh_accept(rng, s::TS, s′::TS) where {TS<:MultinomialTS} = rand(rng) < min(1, exp(s′.ℓw - s.ℓw))

##########################
# Good initial step size #
##########################

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
