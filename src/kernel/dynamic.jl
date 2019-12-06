abstract type DynamicTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed total trajectory length.
###

struct HMCDA{S<:AbstractTrajectorySampler,I<:AbstractIntegrator} <: DynamicTrajectory{I}
    integrator  ::  I
    λ           ::  AbstractFloat
end

Base.show(io::IO, τ::HMCDA{S,I}) where {I,S<:LastTS} =
    print(io, "HMCDA{LastTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")
Base.show(io::IO, τ::HMCDA{S,I}) where {I,S<:MultinomialTS} =
    print(io, "HMCDA{MultinomialTS}(integrator=$(τ.integrator), λ=$(τ.n_steps)))")

HMCDA{S}(integrator::I, λ::AbstractFloat) where {S,I} = HMCDA{S,I}(integrator, λ)
HMCDA(args...) = HMCDA{LastTS}(args...) # default HMCDA using last point from trajectory

function transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    τ::HMCDA{S},
    h::Hamiltonian,
    z::PhasePoint
) where {S}
    # Create the corresponding static τ
    n_steps = max(1, floor(Int, τ.λ / nom_step_size(τ.integrator)))
    static_τ = HMC{S}(τ.integrator, n_steps)
    return transition(rng, static_τ, h, z)
end

###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

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

combine(::ClassicNoUTurn, ::ClassicNoUTurn) = ClassicNoUTurn()
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

Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:SliceTS,C<:ClassicNoUTurn} =
    print(io, "NUTS{SliceTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:SliceTS,C<:GeneralisedNoUTurn} =
    print(io, "NUTS{SliceTS,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:MultinomialTS,C<:ClassicNoUTurn} =
    print(io, "NUTS{MultinomialTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{S,C,I,F}) where {I,F,S<:MultinomialTS,C<:GeneralisedNoUTurn} =
    print(io, "NUTS{MultinomialTS,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")

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
    NUTS(args...) = NUTS{MultinomialTS,GeneralisedNoUTurn}(args...)

Create an instance for the No-U-Turn sampling algorithm
with multinomial sampling and original no U-turn criterion.

Below is the doc for NUTS{S,C}.

$NUTS_DOCSTR
"""
NUTS(args...) = NUTS{MultinomialTS,GeneralisedNoUTurn}(args...)

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
    s::SliceTS,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = Termination(false, !(s.ℓu < nt.Δ_max + -H′))

"""
Check termination of a Hamiltonian trajectory.
"""
Termination(
    s::MultinomialTS,
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
    ΔH_max  # energy in tree with largest absolute different from initial energy
end

"""
    maxabs(a, b)

Return the value with the largest absolute value.
"""
@inline maxabs(a, b) = abs(a) > abs(b) ? a : b

"""
    combine(treeleft::BinaryTree, treeright::BinaryTree)

Merge a left tree `treeleft` and a right tree `treeright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
combine(treeleft::BinaryTree, treeright::BinaryTree) =
    BinaryTree(treeleft.zleft, treeright.zright, combine(treeleft.c, treeright.c), treeleft.sum_α + treeright.sum_α, treeleft.nα + treeright.nα, maxabs(treeleft.ΔH_max, treeright.ΔH_max))

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
        termination = termination * termination′ * isterminated(h, tree)
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
        stat(τ.integrator)
    )

    return Transition(zcand, tstat)
end
