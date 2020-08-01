###
### Advanced HMC implementation with (adaptive) dynamic trajectory length.
###

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

# TODO: Remove this method
"""
    transition(τ, h::Hamiltonian, z::PhasePoint)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit fallback function for `transition(GLOBAL_RNG, τ, h, z)`
"""
function transition(τ, h::Hamiltonian, z::PhasePoint)
    return transition(GLOBAL_RNG, τ, h, z)
end

##
## Variants of no-U-turn criteria
##

ClassicNoUTurn(::PhasePoint) = ClassicNoUTurn()

NoUTurn(z::PhasePoint) = NoUTurn(z.r)

StrictNoUTurn(z::PhasePoint) = StrictNoUTurn(z.r)

combine(::ClassicNoUTurn, ::ClassicNoUTurn) = ClassicNoUTurn()
combine(cleft::T, cright::T) where {T<:NoUTurn} = T(cleft.rho + cright.rho)
combine(cleft::T, cright::T) where {T<:StrictNoUTurn} = T(cleft.rho + cright.rho)


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
}
    integrator      ::  I
    max_depth       ::  Int
    Δ_max           ::  F
end

function Base.show(io::IO, τ::NUTS{<:SliceTS, <:ClassicNoUTurn})
    print(io, "NUTS{SliceTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:SliceTS, <:NoUTurn})
    print(io, "NUTS{SliceTS,Generalised}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:MultinomialTS, <:ClassicNoUTurn})
    print(io, "NUTS{MultinomialTS}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
end
function Base.show(io::IO, τ::NUTS{<:MultinomialTS, <:NoUTurn})
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
    NUTS(args...) = NUTS{MultinomialTS,NoUTurn}(args...)

Create an instance for the No-U-Turn sampling algorithm
with multinomial sampling and original no U-turn criterion.

Below is the doc for NUTS{S,C}.

$NUTS_DOCSTR
"""
NUTS(args...) = NUTS{MultinomialTS, NoUTurn}(args...)

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
    isterminated(h::Hamiltonian, t::BinaryTree{<:NoUTurn})

Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion.

Ref: https://arxiv.org/abs/1701.02434
"""
function isterminated(h::Hamiltonian, t::BinaryTree{<:NoUTurn})
    rho = t.c.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

"""
    isterminated(
        h::Hamiltonian, t::T, tleft::T, tright::T
    ) where {T<:BinaryTree{<:StrictNoUTurn}}

Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
using the generalised no-U-turn criterion with additional U-turn checks.

Ref: https://arxiv.org/abs/1701.02434 https://github.com/stan-dev/stan/pull/2800
"""
function isterminated(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictNoUTurn}}
    # Classic generalised U-turn check
    t_generalised = BinaryTree(
        t.zleft,
        t.zright,
        NoUTurn(t.c.rho),
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
    ) where {T<:BinaryTree{<:StrictNoUTurn}}

Do a U-turn check between the leftmost phase point of `t` and the leftmost 
phase point of `tright`, the right subtree.
"""
function check_left_subtree(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictNoUTurn}}
    rho = tleft.c.rho + tright.zleft.r
    s = generalised_uturn_criterion(rho, ∂H∂r(h, t.zleft.r), ∂H∂r(h, tright.zleft.r))
    return Termination(s, false)
end

"""
    check_left_subtree(
        h::Hamiltonian, t::T, tleft::T, tright::T
    ) where {T<:BinaryTree{<:StrictNoUTurn}}

Do a U-turn check between the rightmost phase point of `t` and the rightmost
phase point of `tleft`, the left subtree.
"""
function check_right_subtree(
    h::Hamiltonian, t::T, tleft::T, tright::T
) where {T<:BinaryTree{<:StrictNoUTurn}}
    rho = tleft.zright.r + tright.c.rho
    s = generalised_uturn_criterion(rho, ∂H∂r(h, tleft.zright.r), ∂H∂r(h, t.zright.r))
    return Termination(s, false)
end

function generalised_uturn_criterion(rho, p_sharp_minus, p_sharp_plus)
    return (dot(rho, p_sharp_minus) <= 0) || (dot(rho, p_sharp_plus) <= 0)
end

function isterminated(h::Hamiltonian, t::BinaryTree{T}, args...) where {T<:Union{ClassicNoUTurn, NoUTurn}}
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

mh_accept(rng::AbstractRNG, s::SliceTS, s′::SliceTS) = rand(rng) < min(1, s′.n / s.n)

function mh_accept(rng::AbstractRNG, s::MultinomialTS, s′::MultinomialTS)
    return rand(rng) < min(1, exp(s′.ℓw - s.ℓw))
end
