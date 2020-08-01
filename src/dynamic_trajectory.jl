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

############################
# TODO: Remove this method #
############################

"""
    transition(τ, h::Hamiltonian, z::PhasePoint)

Make a MCMC transition from phase point `z` using the trajectory `τ` under Hamiltonian `h`.

NOTE: This is a RNG-implicit fallback function for `transition(GLOBAL_RNG, τ, h, z)`
"""
function transition(τ, h::Hamiltonian, z::PhasePoint)
    return transition(GLOBAL_RNG, τ, h, z)
end

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
    @unpack integrator, term_criterion = τ
    H0 = energy(z0)
    tree = BinaryTree(z0, z0, term_criterion, zero(H0), zero(Int), zero(H0))
    sampler = TS(rng, z0)
    termination = Termination(false, false)
    zcand = z0
    j = 0
    while !isterminated(termination) && j < term_criterion.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            tree′, sampler′, termination′ = build_tree(rng, integrator, term_criterion, h, tree.zleft, sampler, v, j, H0)
            treeleft, treeright = tree′, tree
        else
            # Create a tree with depth `j` on the right
            tree′, sampler′, termination′ = build_tree(rng, integrator, term_criterion, h, tree.zright, sampler, v, j, H0)
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
        termination = termination * termination′ * isterminated(term_criterion, h, tree, treeleft, treeright)
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
