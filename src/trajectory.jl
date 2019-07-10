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

#################################
### Hong's abstraction starts ###
#################################

###
### Create a `Termination` type for each `Trajectory` type, e.g. HMC, NUTS etc.
### Merge all `Trajectory` types, and make `transition` dispatch on `Termination`,
### such that we can overload `transition` for different HMC samplers.
### NOTE:  stopping creteria, max_depth::Int, Δ_max::AbstractFloat, n_steps, λ
###

"""
Abstract type for termination.
"""
abstract type AbstractTermination end

# Termination type for HMC and HMCDA
struct StaticTermination{D<:AbstractFloat} <: AbstractTermination
    n_steps :: Int
    Δ_max :: D
end

# NoUTurnTermination
struct NoUTurnTermination{D<:AbstractFloat} <: AbstractTermination
    max_depth :: Int
    Δ_max :: D
    # TODO: add other necessary fields for No-U-Turn stopping creteria.
end

struct Trajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I}
    integrator :: I
    n_steps :: Int # Counter for total leapfrog steps already applied.
    Δ :: AbstractFloat # Current hamiltonian energy minus starting hamiltonian energy
    # TODO: replace all ``*Trajectory` types with `Trajectory`.
    # TODO: add turn statistic, divergent statistic, proposal statistic
end

isterminated(
    x::StaticTermination,
    τ::Trajectory
) = τ.n_steps >= x.n_steps || τ.Δ >= x.Δ_max

# Combine trajectories, e.g. those created by the build_tree algorithm.
#  NOTE: combine proposal (via slice/multinomial sampling), combine turn statistic,
#       and combine divergent statistic.
combine_trajectory(τ′::Trajectory, τ′′::Trajectory) = nothing # To-be-implemented.

## TODO: move slice variable `logu` into `Trajectory`?
combine_proposal(τ′::Trajectory, τ′′::Trajectory) = nothing # To-be-implemented.
combine_turn(τ′::Trajectory, τ′′::Trajectory) = nothing # To-be-implemented.
combine_divergence(τ′::Trajectory, τ′′::Trajectory) = nothing # To-be-implemented.

###############################
### Hong's abstraction ends ###
###############################

transition(
    τ::Trajectory{I},
    h::Hamiltonian,
    z::PhasePoint,
    t::T
) where {I<:AbstractIntegrator,T<:AbstractTermination} = nothing


###
### Standard HMC implementation with fixed leapfrog step numbers.
###
struct StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end

"""
Create a `StaticTrajectory` with a new integrator
"""
function (τ::StaticTrajectory)(integrator::AbstractIntegrator)
    return StaticTrajectory(integrator, τ.n_steps)
end

function transition(
    rng::AbstractRNG,
    τ::StaticTrajectory,
    h::Hamiltonian,
    z::PhasePoint
) where {T<:Real}
    z′ = step(τ.integrator, h, z, τ.n_steps)
    # Accept via MH criteria
    is_accept, α = mh_accept_ratio(rng, -neg_energy(z), -neg_energy(z′))
    if is_accept
        z = PhasePoint(z′.θ, -z′.r, z′.ℓπ, z′.ℓκ)
    end
    return z, α
end

abstract type DynamicTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I} end

###
### Standard HMC implementation with fixed total trajectory length.
###
struct HMCDA{I<:AbstractIntegrator} <: DynamicTrajectory{I}
    integrator  ::  I
    λ           ::  AbstractFloat
end

"""
Create a `HMCDA` with a new integrator
"""
function (τ::HMCDA)(integrator::AbstractIntegrator)
    return HMCDA(integrator, τ.λ)
end

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
abstract type AbstractTreeSampler end

"""
Slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.
"""
struct SliceTreeSampler{F<:AbstractFloat} <: AbstractTreeSampler
    logu    ::  F     # slice variable in log space
    n       ::  Int   # number of acceptable candicates, i.e. those with prob larger than slice variable u
end

"""
Multinomial sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.
"""
struct MultinomialTreeSampler{F<:AbstractFloat} <: AbstractTreeSampler
    w       ::  F     # total energy for the given tree, i.e. sum of energy of all leaves
end

"""
Slice sampler for the starting single leaf tree.
Slice variable is initialized.
"""
SliceTreeSampler(rng::AbstractRNG, H::AbstractFloat) = SliceTreeSampler(log(rand(rng)) - H, 1)

"""
Multinomial sampler for the starting single leaf tree.
Tree weight is just the probability of the only leave.
"""
MultinomialTreeSampler(rng::AbstractRNG, H::AbstractFloat) = MultinomialTreeSampler(exp(-H))

"""
Create a slice sampler for a single leaf tree:
- the slice variable is copied from the passed-in sampler `s` and
- the number of acceptable candicates is computed by comparing the slice variable against the current energy.
"""
makebase(s::SliceTreeSampler, H::AbstractFloat) = SliceTreeSampler(s.logu, (s.logu <= -H) ? 1 : 0)

"""
Create a multinomial sampler for a single leaf tree:
- the tree weight is just the probability of the only leave.
"""
makebase(s::MultinomialTreeSampler, H::AbstractFloat) = MultinomialTreeSampler(exp(-H))

combine(s1::SliceTreeSampler, s2::SliceTreeSampler) = SliceTreeSampler(s1.logu, s1.n + s2.n)
combine(s1::MultinomialTreeSampler, s2::MultinomialTreeSampler) = MultinomialTreeSampler(s1.w + s2.w)

"""
Dynamic trajectory HMC using the no-U-turn termination criteria algorithm.
"""
struct NUTS{
    I<:AbstractIntegrator,
    F<:AbstractFloat,
    S<:AbstractTreeSampler
} <: DynamicTrajectory{I}
    integrator  ::  I
    max_depth   ::  Int
    Δ_max       ::  F
    samplerType ::  Type{S}
end

"""
Helper dictionary used to allow users pass symbol keyword argument
to create NUTS with different sampling algorithm.
"""
const SUPPORTED_TREE_SAMPLING = Dict(:slice => SliceTreeSampler, :multinomial => MultinomialTreeSampler)
const DEFAULT_TREE_SAMPLING = :multinomial

"""
    NUTS(
        integrator::AbstractIntegrator,
        max_depth::Int=10,
        Δ_max::AbstractFloat=1000.0;
        sampling::Symbol=:multinomial
    )

Create an instance for the No-U-Turn sampling algorithm.
"""
function NUTS(
    integrator::AbstractIntegrator,
    max_depth::Int=10,
    Δ_max::AbstractFloat=1000.0;
    sampling::Symbol=DEFAULT_TREE_SAMPLING
)
    @assert sampling in keys(SUPPORTED_TREE_SAMPLING) "NUTS only supports the following sampling methods: $(keys(SUPPORTED_TREE_SAMPLING))"
    return NUTS(integrator, max_depth, Δ_max, SUPPORTED_TREE_SAMPLING[sampling])
end
@info "Default NUTS tree sampling method is set to $DEFAULT_TREE_SAMPLING."

"""
Create a new No-U-Turn sampling algorithm with a new integrator.
"""
function (nuts::NUTS)(integrator::AbstractIntegrator)
    return NUTS(integrator, nuts.max_depth, nuts.Δ_max, nuts.samplerType)
end

###
### The doubling tree algorithm for expanding trajectory.
###

"""
A full binary tree trajectory with only necessary leaves and information stored.
"""
struct FullBinaryTree{S<:AbstractTreeSampler}
    zleft       # left most leaf node
    zright      # right most leaf node
    zcand       # candidate leaf node
    sampler::S  # condidate sampler
    s           # termination stats, i.e. 0 means termination and 1 means continuation
    α           # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα          # total # of leap frog steps, i.e. phase points in a trajectory
end

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
"""
function isUturn(h::Hamiltonian, zleft::PhasePoint, zright::PhasePoint)
    θdiff = zright.θ - zleft.θ
    return (dot(θdiff, ∂H∂r(h, zleft.r)) >= 0 ? 1 : 0) * (dot(θdiff, ∂H∂r(h, zright.r)) >= 0 ? 1 : 0)
end


"""
    merge(h::Hamiltonian, tleft::FullBinaryTree, tright::FullBinaryTree)

Merge a left tree `tleft` and a right tree `tright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
function merge(
    h::Hamiltonian,
    tleft::FullBinaryTree,
    tright::FullBinaryTree;
    rng::AbstractRNG = GLOBAL_RNG
)
    zleft = tleft.zleft
    zright = tright.zright
    zcand = combine(tleft, tright; rng=rng)
    sampler = combine(tleft.sampler, tright.sampler)
    s = tleft.s * tright.s * isUturn(h, zleft, zright)
    return FullBinaryTree(zleft, zright, zcand, sampler, s, tright.α + tright.α, tright.nα + tright.nα)
end

"""
Check whether the Hamiltonian trajectory has diverged.
"""
iscontinued(
    s::SliceTreeSampler,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = (s.logu < nt.Δ_max + -H′) ? 1 : 0
iscontinued(
    s::MultinomialTreeSampler,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = (-H0 < nt.Δ_max + -H′) ? 1 : 0

"""
Sample a condidate point form two trees (`tleft` and `tright`) under slice sampling.
"""
function combine(
    tleft::FullBinaryTree{SliceTreeSampler{F}},
    tright::FullBinaryTree{SliceTreeSampler{F}};
    rng::AbstractRNG = GLOBAL_RNG
) where {F<:AbstractFloat}
    return rand(rng) < tleft.sampler.n / (tleft.sampler.n + tright.sampler.n) ? tleft.zcand : tright.zcand
end

"""
Sample a condidate point form two trees (`tleft` and `tright`) under multinomial sampling.
"""
function combine(
    tleft::FullBinaryTree{MultinomialTreeSampler{F}},
    tright::FullBinaryTree{MultinomialTreeSampler{F}};
    rng::AbstractRNG = GLOBAL_RNG
) where {F<:AbstractFloat}
    return rand(rng) < tleft.sampler.w / (tleft.sampler.w + tright.sampler.w) ? tleft.zcand : tright.zcand
end

"""
Recursivly build a tree for a given depth `j`.
"""
function build_tree(
    rng::AbstractRNG,
    nt::NUTS{I,F,S},
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractTreeSampler,
    v::Int,
    j::Int,
    H0::AbstractFloat
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTreeSampler}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = -neg_energy(z′)
        basesampler = makebase(sampler, H′)
        s′ = iscontinued(basesampler, nt, H0, H′)
        α′ = exp(min(0, H0 - H′))
        return FullBinaryTree(z′, z′, z′, basesampler, s′, α′, 1)
    else
        # Recursion - build the left and right subtrees.
        t′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if t′.s == 1
            # Expand left
            if v == -1
                t′′ = build_tree(rng, nt, h, t′.zleft, sampler, v, j - 1, H0) # left tree
                tleft, tright = t′′, t′
            # Expand right
            else
                t′′ = build_tree(rng, nt, h, t′.zright, sampler, v, j - 1, H0) # right tree
                tleft, tright = t′, t′′
            end
            t′ = merge(h, tleft, tright; rng=rng)
        end
        return t′
    end
end

mh_accept(
    rng::AbstractRNG,
    s::SliceTreeSampler,
    s′::SliceTreeSampler
) = rand(rng) < min(1, s′.n / s.n)
mh_accept(
    rng::AbstractRNG,
    s::MultinomialTreeSampler,
    s′::MultinomialTreeSampler
) = rand(rng) < min(1, s′.w / s.w)

function transition(
    rng::AbstractRNG,
    nt::NUTS{I,F,S},
    h::Hamiltonian,
    z0::PhasePoint
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTreeSampler}
    H0 = -neg_energy(z0)

    zleft = z0; zright = z0; zcand = z0; j = 0; s = 1; sampler = S(rng, H0)

    local t
    while s == 1 && j <= nt.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            t = build_tree(rng, nt, h, zleft, sampler, v, j, H0)
            zleft = t.zleft
        else
            # Create a tree with depth `j` on the right
            t = build_tree(rng, nt, h, zright, sampler, v, j, H0)
            zright = t.zright
        end
        # Perform a MH step if not terminated
        if t.s == 1 && mh_accept(rng, sampler, t.sampler)
            zcand = t.zcand
        end
        # Combine the sampler from the proposed tree and the current tree
        sampler = combine(sampler, t.sampler)
        # Detect termination
        s = s * t.s * isUturn(h, zleft, zright)
        # Increment tree depth
        j = j + 1
    end

    return zcand, t.α / t.nα
end

"""
A single Hamiltonian integration step.

NOTE: this function is intended to be used in `find_good_eps` only.
"""
function A(rng, h, z, ϵ)
    z′ = step(Leapfrog(ϵ), h, z)
    H′ = -neg_energy(z′)
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
    ϵ′ = ϵ = 0.1
    a_min, a_cross, a_max = 0.25, 0.5, 0.75 # minimal, crossing, maximal accept ratio
    d = 2.0
    # Create starting phase point
    r = rand(rng, h.metric) # sample momentum variable
    z = phasepoint(h, θ, r)
    H = -neg_energy(z)

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

update(
    h::Hamiltonian,
    τ::AbstractProposal,
    dpc::Adaptation.AbstractPreconditioner
) = h(getM⁻¹(dpc)), τ

update(
    h::Hamiltonian,
    τ::AbstractProposal,
    da::NesterovDualAveraging
) = h, τ(τ.integrator(getϵ(da)))

update(
    h::Hamiltonian,
    τ::AbstractProposal,
    ca::Union{Adaptation.NaiveHMCAdaptor, Adaptation.StanHMCAdaptor}
) = h(getM⁻¹(ca.pc)), τ(τ.integrator(getϵ(ca.ssa)))

function update(
    h::Hamiltonian,
    θ::AbstractVector{T}
) where {T<:Real}
    metric = h.metric
    if length(metric) != length(θ)
        metric = metric(length(θ))
        h = h(getM⁻¹(Preconditioner(metric)))
    end
    return h
end
