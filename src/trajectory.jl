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
    is_accept, α = mh_accept_ratio(rng, -neg_energy(z), -neg_energy(z′))
    if is_accept
        z = PhasePoint(z′.θ, -z′.r, z′.ℓπ, z′.ℓκ)
    end
    stat = (
        step_size=τ.integrator.ϵ, 
        n_steps=τ.n_steps, 
        is_accept=is_accept, 
        acceptance_rate=α, 
        log_density=z.ℓπ.value, 
        hamiltonian_energy=-neg_energy(z), 
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
abstract type AbstractTreeSampler end

"""
Slice sampler carried during the building of the tree.
It contains the slice variable and the number of acceptable condidates in the tree.
"""
struct SliceTreeSampler{F<:AbstractFloat} <: AbstractTreeSampler
    logu    ::  F     # slice variable in log space
    n       ::  Int   # number of acceptable candicates, i.e. those with prob larger than slice variable u
end

Base.show(io::IO, s::SliceTreeSampler) = print(io, "SliceTreeSampler(logu=$(s.logu), n=$(s.n))")

"""
Multinomial sampler carried during the building of the tree.
It contains the weight of the tree, defined as the total probabilities of the leaves.
"""
struct MultinomialTreeSampler{F<:AbstractFloat} <: AbstractTreeSampler
    logw    ::  F     # total energy for the given tree, i.e. sum of energy of all leaves
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
MultinomialTreeSampler(rng::AbstractRNG, H::AbstractFloat) = MultinomialTreeSampler(-H)

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
makebase(s::MultinomialTreeSampler, H::AbstractFloat) = MultinomialTreeSampler(-H)

combine(s1::SliceTreeSampler, s2::SliceTreeSampler) = SliceTreeSampler(s1.logu, s1.n + s2.n)
combine(s1::MultinomialTreeSampler, s2::MultinomialTreeSampler) = MultinomialTreeSampler(logaddexp(s1.logw, s2.logw))

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
Base.show(io::IO, τ::NUTS{I,F,S}) where {I,F,S<:SliceTreeSampler} = 
    print(io, "NUTS{Slice}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")
Base.show(io::IO, τ::NUTS{I,F,S}) where {I,F,S<:MultinomialTreeSampler} = 
    print(io, "NUTS{Multinomial}(integrator=$(τ.integrator), max_depth=$(τ.max_depth)), Δ_max=$(τ.Δ_max))")

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
    @assert sampling in keys(SUPPORTED_TREE_SAMPLING) "NUTS only supports the following
        sampling methods: $(keys(SUPPORTED_TREE_SAMPLING))"
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
A full binary tree trajectory with only necessary leaves and information stored.
"""
struct FullBinaryTree{S<:AbstractTreeSampler}
    zleft       # left most leaf node
    zright      # right most leaf node
    zcand       # candidate leaf node
    sampler::S  # condidate sampler
    termination # termination reasons
    α           # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα          # total # of leap frog steps, i.e. phase points in a trajectory
end

"""
Detect U turn for two phase points (`zleft` and `zright`) under given Hamiltonian `h`
"""
function isturn(h::Hamiltonian, zleft::PhasePoint, zright::PhasePoint)
    θdiff = zright.θ - zleft.θ
    s = (dot(θdiff, ∂H∂r(h, zleft.r)) >= 0 ? 1 : 0) * (dot(θdiff, ∂H∂r(h, zright.r)) >= 0 ? 1 : 0)
    return Termination(s == 0, false)
end
# function isturn(h::Hamiltonian, zleft::PhasePoint, zright::PhasePoint)
#     θdiff = zright.θ - zleft.θ
#     s = (dot(-θdiff, ∂H∂r(h, zleft.r)) < 0) && (dot(θdiff, ∂H∂r(h, zright.r)) < 0)
#     return Termination(s, false)
# end

"""
Check termination of a Hamiltonian trajectory.
"""
isterminated(
    s::SliceTreeSampler,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = Termination(false, !(s.logu < nt.Δ_max + -H′))
isterminated(
    s::MultinomialTreeSampler,
    nt::NUTS,
    H0::F,
    H′::F
) where {F<:AbstractFloat} = Termination(false, !(-H0 < nt.Δ_max + -H′))

"""
    combine(h::Hamiltonian, tleft::FullBinaryTree, tright::FullBinaryTree)

Merge a left tree `tleft` and a right tree `tright` under given Hamiltonian `h`,
then draw a new candidate sample and update related statistics for the resulting tree.
"""
function combine(
    rng::AbstractRNG,
    h::Hamiltonian,
    tleft::FullBinaryTree,
    tright::FullBinaryTree
)
    zleft = tleft.zleft
    zright = tright.zright
    zcand = combine(rng, tleft, tright)
    sampler = combine(tleft.sampler, tright.sampler)
    termination = tleft.termination * tright.termination * isturn(h, zleft, zright)
    return FullBinaryTree(zleft, zright, zcand, sampler, termination, tright.α + tright.α, tright.nα + tright.nα)
end

"""
Sample a condidate point form two trees (`tleft` and `tright`) under slice sampling.
"""
function combine(
    rng::AbstractRNG,
    tleft::FullBinaryTree{SliceTreeSampler{F}},
    tright::FullBinaryTree{SliceTreeSampler{F}}
) where {F<:AbstractFloat}
    return rand(rng) < tleft.sampler.n / (tleft.sampler.n + tright.sampler.n) ? tleft.zcand : tright.zcand
end

"""
Sample a condidate point form two trees (`tleft` and `tright`) under multinomial sampling.
"""
function combine(
    rng::AbstractRNG,
    tleft::FullBinaryTree{MultinomialTreeSampler{F}},
    tright::FullBinaryTree{MultinomialTreeSampler{F}}
) where {F<:AbstractFloat}
    return rand(rng) < exp(tleft.sampler.logw - logaddexp(tleft.sampler.logw, tright.sampler.logw)) ? tleft.zcand : tright.zcand
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
        termination = isterminated(basesampler, nt, H0, H′)
        α′ = exp(min(0, H0 - H′))
        return FullBinaryTree(z′, z′, z′, basesampler, termination, α′, 1)
    else
        # Recursion - build the left and right subtrees.
        t′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !isterminated(t′.termination)
            # Expand left
            if v == -1
                t′′ = build_tree(rng, nt, h, t′.zleft, sampler, v, j - 1, H0) # left tree
                tleft, tright = t′′, t′
            # Expand right
            else
                t′′ = build_tree(rng, nt, h, t′.zright, sampler, v, j - 1, H0) # right tree
                tleft, tright = t′, t′′
            end
            t′ = combine(rng, h, tleft, tright)
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
) = rand(rng) < exp(min(0, s′.logw - s.logw))

function transition(
    rng::AbstractRNG,
    τ::NUTS{I,F,S},
    h::Hamiltonian,
    z0::PhasePoint
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractTreeSampler}
    H0 = -neg_energy(z0)

    zleft = z0; zright = z0; z = z0; 
    j = 0; termination = Termination(false, false); sampler = S(rng, H0)

    local t
    while !isterminated(termination) && j <= τ.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            t = build_tree(rng, τ, h, zleft, sampler, v, j, H0)
            zleft = t.zleft
        else
            # Create a tree with depth `j` on the right
            t = build_tree(rng, τ, h, zright, sampler, v, j, H0)
            zright = t.zright
        end
        # Perform a MH step if not terminated
        if !isterminated(t.termination) && mh_accept(rng, sampler, t.sampler)
            z = t.zcand
        end
        # Combine the sampler from the proposed tree and the current tree
        sampler = combine(sampler, t.sampler)
        # Detect termination
        termination = termination * t.termination * isturn(h, zleft, zright)
        # Increment tree depth
        j = j + 1
    end

    stat = (
        step_size=τ.integrator.ϵ, 
        n_steps=2^j, 
        is_accept=true, 
        acceptance_rate=t.α / t.nα, 
        log_density=z.ℓπ.value, 
        hamiltonian_energy=-neg_energy(z), 
        tree_depth=j, 
        numerical_error=termination.numerical,
       )
    return z, stat
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
