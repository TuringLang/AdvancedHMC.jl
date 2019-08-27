####
#### Implementation for Hamiltonian dynamics trajectories
####
#### Developers' Notes
####
#### Not all functions that use `rng` requires a callback function with `GLOBAL_RNG`
#### as default. In short, only those exported to other libries need such a callback
#### function. Internal uses shall always use the explict `rng` version. (Kai Xu 6/Jul/19)

"""
Abstract Markov chain Monte Carlo proposal.
"""
abstract type AbstractProposal end

"""
Hamiltonian dynamics numerical simulation trajectories.
"""
abstract type AbstractTrajectory{I<:AbstractIntegrator} <: AbstractProposal end

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
### Standard HMC implementation with fixed leapfrog step numbers.
###
struct StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end
Base.show(io::IO, τ::StaticTrajectory) = print(io, "StaticTrajectory(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

function transition(
    rng::AbstractRNG,
    τ::StaticTrajectory,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    z′ = step(τ.integrator, h, z, τ.n_steps)
    # Accept via MH criteria
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    if is_accept
        # Reverse momentum variable to preserve reversibility
        z = PhasePoint(z′.θ, -z′.r, z′.ℓπ, z′.ℓκ)
    end
    stat = (
        step_size=τ.integrator.ϵ, 
        n_steps=τ.n_steps, 
        is_accept=is_accept, 
        acceptance_rate=α, 
        log_density=z.ℓπ.value, 
        hamiltonian_energy=energy(z), 
       )
    return z, stat
end

abstract type DynamicTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed total trajectory length.
###
struct HMCDA{I<:AbstractIntegrator} <: DynamicTrajectory{I}
    integrator  ::  I
    λ           ::  AbstractFloat
end
Base.show(io::IO, τ::HMCDA) = print(io, "HMCDA(integrator=$(τ.integrator), λ=$(τ.λ)))")

function transition(
    rng::AbstractRNG,
    τ::HMCDA,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    # Create the corresponding static τ
    n_steps = max(1, floor(Int, τ.λ / τ.integrator.ϵ))
    static_τ = StaticTrajectory(τ.integrator, n_steps)
    return transition(rng, static_τ, h, z)
end


###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

##
## Slice and multinomial sampling for trajectories.
##

"""
Sampler carried during the building of the tree.
"""
abstract type AbstractTrajectorySampler end

"""
Slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.
"""
struct Slice{F<:AbstractFloat} <: AbstractTrajectorySampler
    zcand   ::  PhasePoint
    ℓu      ::  F     # slice variable in log space
    n       ::  Int   # number of acceptable candicates, i.e. those with prob larger than slice variable u
end

Base.show(io::IO, s::Slice) = print(io, "Slice(ℓu=$(s.ℓu), n=$(s.n))")

"""
Multinomial sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.
"""
struct Multinomial{F<:AbstractFloat} <: AbstractTrajectorySampler
    zcand   ::  PhasePoint
    ℓw      ::  F     # total energy for the given tree, i.e. sum of energy of all leaves
end

"""
Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
Slice(rng::AbstractRNG, z0::PhasePoint) = Slice(z0, log(rand(rng)) - energy(z0), 1)

"""
Multinomial sampler for the starting single leaf tree.
(Log) weights for leaf nodes are their (unnormalised) Hamiltonian energies.

Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp#L226
"""
Multinomial(rng::AbstractRNG, z0::PhasePoint) = Multinomial(z0, zero(energy(z0)))

"""
Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
Slice(s::Slice, H0::AbstractFloat, zcand::PhasePoint) = 
    Slice(zcand, s.ℓu, (s.ℓu <= -energy(zcand)) ? 1 : 0)

"""
Multinomial sampler for a trajectory consisting only a leaf node.
- tree weight is the (unnormalised) energy of the leaf.
"""
Multinomial(s::Multinomial, H0::AbstractFloat, zcand::PhasePoint) = 
    Multinomial(zcand, H0 - energy(zcand))

function combine(rng::AbstractRNG, s1::Slice, s2::Slice)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    zcand = rand(rng) < s1.n / n ? s1.zcand : s2.zcand
    Slice(zcand, s1.ℓu, n)
end

function combine(zcand::PhasePoint, s1::Slice, s2::Slice)
    @assert s1.ℓu == s2.ℓu "Cannot combine two slice sampler with different slice variable"
    n = s1.n + s2.n
    Slice(zcand, s1.ℓu, n)
end

function combine(rng::AbstractRNG, s1::Multinomial, s2::Multinomial)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    zcand = rand(rng) < exp(s1.ℓw - ℓw) ? s1.zcand : s2.zcand
    Multinomial(zcand, ℓw)
end

function combine(zcand::PhasePoint, s1::Multinomial, s2::Multinomial)
    ℓw = logaddexp(s1.ℓw, s2.ℓw)
    Multinomial(zcand, ℓw)
end

mh_accept(
    rng::AbstractRNG,
    s::Slice,
    s′::Slice
) = rand(rng) < min(1, s′.n / s.n)

mh_accept(
    rng::AbstractRNG,
    s::Multinomial,
    s′::Multinomial
) = rand(rng) < min(1, exp(s′.ℓw - s.ℓw))

##
## Variants of no-U-turn criteria
##

abstract type AbstractTerminationCriterion end

struct ClassicNoUTurn <: AbstractTerminationCriterion end

ClassicNoUTurn(::PhasePoint) = ClassicNoUTurn()

struct GeneralisedNoUTurn{T<:AbstractVector{<:Real}} <: AbstractTerminationCriterion
    rho::T
end

GeneralisedNoUTurn(z::PhasePoint) = GeneralisedNoUTurn(z.r)

combine(cleft::T, cright::T) where {T<:ClassicNoUTurn} = T()
combine(cleft::T, cright::T) where {T<:GeneralisedNoUTurn} = T(cleft.rho + cright.rho)


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

Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:Slice,C<:ClassicNoUTurn} = 
    print(io, "NUTS{Slice}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:Slice,C<:GeneralisedNoUTurn} = 
    print(io, "NUTS{Slice,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:Multinomial,C<:ClassicNoUTurn} = 
    print(io, "NUTS{Multinomial}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:Multinomial,C<:GeneralisedNoUTurn} = 
    print(io, "NUTS{Multinomial,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")

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
    Δ_max::F=1000.0
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    return NUTS{S,C,I,F}(integrator, max_depth, Δ_max)
end

"""
    NUTS(args...) = NUTS{Multinomial,GeneralisedNoUTurn}(args...)

Create an instance for the No-U-Turn sampling algorithm
with multinomial sampling and original no U-turn criterion.

Below is the doc for NUTS{S,C}.

$NUTS_DOCSTR
"""
NUTS(args...) = NUTS{Multinomial,GeneralisedNoUTurn}(args...)

###
### The doubling tree algorithm for expanding trajectory.
###

"""
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
Check termination of a Hamiltonian trajectory.
"""
Termination(
    s::Slice,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = Termination(false, !(s.ℓu < nt.Δ_max + -H′))

"""
Check termination of a Hamiltonian trajectory.
"""
Termination(
    s::Multinomial,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = Termination(false, !(-H0 < nt.Δ_max + -H′))

"""
A full binary tree trajectory with only necessary leaves and information stored.
"""
struct BinaryTree{C<:AbstractTerminationCriterion}
    zleft   # left most leaf node
    zright  # right most leaf node
    c::C    # termination criterion
    sum_α   # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα      # total # of leap frog steps, i.e. phase points in a trajectory
end

"""
    combine(treeleft::BinaryTree, treeright::BinaryTree)

Merge a left tree `treeleft` and a right tree `treeright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
combine(treeleft::BinaryTree, treeright::BinaryTree) = 
    BinaryTree(treeleft.zleft, treeright.zright, combine(treeleft.c, treeright.c), treeleft.sum_α + treeright.sum_α, treeleft.nα + treeright.nα)

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the (original) no-U-turn cirterion.

Ref: https://arxiv.org/abs/1111.4246, https://arxiv.org/abs/1701.02434
"""
function isterminated(h::Hamiltonian, t::BinaryTree{C}) where {C<:ClassicNoUTurn}
    # z0 is starting point and z1 is ending point
    z0, z1 = t.zleft, t.zright
    Δθ = z1.θ - z0.θ
    s = (dot(Δθ, ∂H∂r(h, -z0.r)) >= 0) || (dot(-Δθ, ∂H∂r(h, z1.r)) >= 0)
    return Termination(s, false)
end

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion.

Ref: https://arxiv.org/abs/1701.02434
"""
function isterminated(h::Hamiltonian, t::BinaryTree{C}) where {C<:GeneralisedNoUTurn}
    # z0 is starting point and z1 is ending point
    z0, z1 = t.zleft, t.zright
    rho = t.c.rho
    s = (dot(rho, ∂H∂r(h, -z0.r)) >= 0) || (dot(-rho, ∂H∂r(h, z1.r)) >= 0)
    return Termination(s, false)
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
    H0::AbstractFloat
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = energy(z′)
        α′ = exp(min(0, H0 - H′))
        sampler′ = S(sampler, H0, z′)
        return BinaryTree(z′, z′, C(z′), α′, 1), sampler′, Termination(sampler′, nt, H0, H′)
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
            termination′ = termination′ * termination′′ * isterminated(h, tree′)
        end
        return tree′, sampler′, termination′
    end
end

function transition(
    rng::AbstractRNG,
    τ::NUTS{S,C,I,F},
    h::Hamiltonian,
    z0::PhasePoint
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTrajectorySampler,C<:AbstractTerminationCriterion}
    H0 = energy(z0)
    tree = BinaryTree(z0, z0, C(z0), zero(F), zero(Int))
    sampler = S(rng, z0)
    termination = Termination(false, false)
    zcand = z0

    j = 0
    while !isterminated(termination) && j <= τ.max_depth
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
        termination = termination * termination′ * isterminated(h, tree)
    end

    stat = (
        step_size=τ.integrator.ϵ, 
        n_steps=tree.nα, 
        is_accept=true, 
        acceptance_rate=tree.sum_α / tree.nα, 
        log_density=zcand.ℓπ.value, 
        hamiltonian_energy=energy(zcand), 
        tree_depth=j, 
        numerical_error=termination.numerical,
       )
    return zcand, stat
end

"""
A single Hamiltonian integration step.

NOTE: this function is intended to be used in `find_good_eps` only.
"""
function A(rng, h, z, ϵ)
    z′ = step(Leapfrog(ϵ), h, z)
    H′ = energy(z′)
    return z′, H′
end

"""
Find a good initial leap-frog step-size via heuristic search.
"""
function find_good_eps(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100
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
