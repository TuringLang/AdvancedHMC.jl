##
## Hamiltonian dynamics numerical simulation trajectories
##

abstract type AbstractProposal end
abstract type AbstractTrajectory{I<:AbstractIntegrator} <: AbstractProposal end

# Create a callback function for all `AbstractTrajectory`
# without passing random number generator
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
    is_accept, α = mh_accept(rng, -neg_energy(z), -neg_energy(z′))
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

# Types for slice and multinomial sampling; with types the branches for different sampling methods shall be compiled away
abstract type AbstractNUTSSampling end

struct SliceNUTSSampling <: AbstractNUTSSampling end
struct MultinomialNUTSSampling <: AbstractNUTSSampling end
const SUPPORTED_NUTS_SAMPLING = Dict(:slice => SliceNUTSSampling(), :multinomial => MultinomialNUTSSampling())

abstract type AbstractNUTSSampler end

struct SliceNUTSSampler{F<:AbstractFloat} <: AbstractNUTSSampler 
    logu    ::  F     # slice variable in log space
    n       ::  Int   # number of acceptable candicates, i.e. prob is larger than slice variable u
end
struct MultinomialNUTSSampler{F<:AbstractFloat} <: AbstractNUTSSampler
    w       ::  F     # total energy for the given tree, i.e. sum of energy of all leaves
end

combine(s1::SliceNUTSSampler, s2::SliceNUTSSampler) = SliceNUTSSampler(s1.logu, s1.n + s2.n)
combine(s1::MultinomialNUTSSampler, s2::MultinomialNUTSSampler) = MultinomialNUTSSampler(s1.w + s2.w)

"""
Dynamic trajectory HMC using the no-U-turn termination criteria algorithm.
"""
struct NUTS{I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractNUTSSampling} <: DynamicTrajectory{I}
    integrator  ::  I
    max_depth   ::  Int
    Δ_max       ::  F
    sampling    ::  S
end

# Helper function to use default values
function NUTS(
    integrator::AbstractIntegrator, 
    max_depth::Int=10, 
    Δ_max::AbstractFloat=1000.0;
    sampling::Symbol=:multinomial
) 
    @assert sampling in keys(SUPPORTED_NUTS_SAMPLING) "NUTS only supports the following sampling methods: $(keys(SUPPORTED_NUTS_SAMPLING))"
    return NUTS(integrator, max_depth, Δ_max, SUPPORTED_NUTS_SAMPLING[sampling])
end

"""
Create a `NUTS` with a new integrator
"""
function (nuts::NUTS)(integrator::AbstractIntegrator)
    return NUTS(integrator, nuts.max_depth, nuts.Δ_max, nuts.sampling)
end


###
### The doubling tree algorithm for expanding trajectory.
###

struct DoublingTree{S<:AbstractNUTSSampler}
    zleft       # left most leaf node
    zright      # right most leaf node
    zcand       # candidate leaf node
    sampler::S  # condidate sampler
    s           # termination stats, i.e. 0 means termination and 1 means continuation
    α           # MH stats, i.e. sum of MH accept prob for all leapfrog steps
    nα          # total # of leap frog steps, i.e. phase points in a trajectory
end
# TODO: merge DoublingTree and Trajectory

function isUturn(h::Hamiltonian, zleft::PhasePoint, zright::PhasePoint)
    θdiff = zright.θ - zleft.θ
    return (dot(θdiff, ∂H∂r(h, zleft.r)) >= 0 ? 1 : 0) * (dot(θdiff, ∂H∂r(h, zright.r)) >= 0 ? 1 : 0)
end

function sample(rng::AbstractRNG, dtleft::DoublingTree{SliceNUTSSampler{F}}, dtright::DoublingTree{SliceNUTSSampler{F}}) where {F<:AbstractFloat}
    return rand(rng) < dtleft.sampler.n / (dtleft.sampler.n + dtright.sampler.n) ? dtleft.zcand : dtright.zcand
end
function sample(rng::AbstractRNG, dtleft::DoublingTree{MultinomialNUTSSampler{F}}, dtright::DoublingTree{MultinomialNUTSSampler{F}}) where {F<:AbstractFloat}
    return rand(rng) < dtleft.sampler.w / (dtleft.sampler.w + dtright.sampler.w) ? dtleft.zcand : dtright.zcand
end
sample(dtleft::DoublingTree, dtright::DoublingTree) = sample(GLOBAL_RNG, dtleft, dtright)

function merge(rng::AbstractRNG, h::Hamiltonian, dtleft::DoublingTree, dtright::DoublingTree)
    zleft = dtleft.zleft
    zright = dtright.zright
    zcand = sample(rng, dtleft, dtright)
    sampler = combine(dtleft.sampler, dtright.sampler)
    s = dtleft.s * dtright.s * isUturn(h, zleft, zright)
    return DoublingTree(zleft, zright, zcand, sampler, s, dtright.α + dtright.α, dtright.nα + dtright.nα)
end

"""
    merge(h::Hamiltonian, dtleft::DoublingTree, dtright::DoublingTree)

Merge a left tree `dtleft` and a right tree `dtright` under given Hamiltonian `h`.
"""
merge(h::Hamiltonian, dtleft::DoublingTree, dtright::DoublingTree) = merge(GLOBAL_RNG, h, dtleft, dtright)
iscontinued(s::SliceNUTSSampler, nt::NUTS, H::AbstractFloat, H′::AbstractFloat) = (s.logu < nt.Δ_max + -H′) ? 1 : 0
# REVIEW: @Hong can you please double check if the implementation below is correct
iscontinued(s::MultinomialNUTSSampler, nt::NUTS, H::AbstractFloat, H′::AbstractFloat) = (-H < nt.Δ_max + -H′) ? 1 : 0

BaseSampler(s::SliceNUTSSampler, H::AbstractFloat) = SliceNUTSSampler(s.logu, (s.logu <= -H) ? 1 : 0)
BaseSampler(s::MultinomialNUTSSampler, H::AbstractFloat) = MultinomialNUTSSampler(H)

function build_tree(
    rng::AbstractRNG,
    nt::NUTS{I,F,S},
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractNUTSSampler,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractNUTSSampling}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = -neg_energy(z′)
        sampler = BaseSampler(sampler, H′)
        s′ = iscontinued(sampler, nt, H, H′)
        α′ = exp(min(0, H - H′))
        return DoublingTree(z′, z′, z′, sampler, s′, α′, 1)
    else
        # Recursion - build the left and right subtrees.
        dt′ = build_tree(rng, nt, h, z, sampler, v, j - 1, H)
        # Expand tree if not terminated
        if dt′.s == 1
            # Expand left
            if v == -1
                dt′′ = build_tree(rng, nt, h, dt′.zleft, sampler, v, j - 1, H) # left tree
                dtleft, dtright = dt′′, dt′
            # Expand right
            else    
                dt′′ = build_tree(rng, nt, h, dt′.zright, sampler, v, j - 1, H) # right tree
                dtleft, dtright = dt′, dt′′
            end
            dt′ = merge(rng, h, dtleft, dtright)
        end
        return dt′
    end
end

"""
Recursivly build a tree for a given depth `j`.
"""
build_tree(
    nt::NUTS{I,F,S},
    h::Hamiltonian,
    z::PhasePoint,
    sampler::AbstractNUTSSampler,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractNUTSSampling} = build_tree(GLOBAL_RNG, nt, h, z, sampler, v, j, H)

mh_accept(rng::AbstractRNG, s::SliceNUTSSampler, s′::SliceNUTSSampler) = mh_accept(rng, s.n, s′.n)
mh_accept(rng::AbstractRNG, s::MultinomialNUTSSampler, s′::MultinomialNUTSSampler) = mh_accept(rng, s.w, s′.w)
mh_accept(s::AbstractNUTSSampler, s′::AbstractNUTSSampler) = mh_accept(GLOBAL_RNG, s, s′)

InitSampler(rng::AbstractRNG, ::SliceNUTSSampling, H::AbstractFloat) = SliceNUTSSampler(log(rand(rng)) - H, 1)
InitSampler(rng::AbstractRNG, ::MultinomialNUTSSampling, H::AbstractFloat) = MultinomialNUTSSampler(H)
InitSampler(s::AbstractNUTSSampling, H::AbstractFloat) = InitSampler(GLOBAL_RNG, s, H)

function transition(
    rng::AbstractRNG,
    nt::NUTS{I,F,S},
    h::Hamiltonian,
    z::PhasePoint
) where {I<:AbstractIntegrator,F<:AbstractFloat,S<:AbstractNUTSSampling}
    H = -neg_energy(z)

    zleft = z; zright = z; zcand = z; j = 0; s = 1; sampler = InitSampler(rng, nt.sampling, H)

    local dt
    while s == 1 && j <= nt.max_depth
        # Sample a direction; `-1` means left and `1` means right
        v = rand(rng, [-1, 1])
        if v == -1
            # Create a tree with depth `j` on the left
            dt = build_tree(rng, nt, h, zleft, sampler, v, j, H)
            zleft = dt.zleft
        else
            # Create a tree with depth `j` on the right
            dt = build_tree(rng, nt, h, zright, sampler, v, j, H)
            zright = dt.zright
        end
        # Perform a MH step if not terminated
        if dt.s == 1 && mh_accept(rng, sampler, dt.sampler)[1]
            zcand = dt.zcand
        end
        # Combine the sampler from the proposed tree and the current tree
        sampler = combine(sampler, dt.sampler)
        # Detect termination
        s = s * dt.s * isUturn(h, zleft, zright)
        # Increment tree depth
        j = j + 1
    end

    return zcand, dt.α / dt.nα
end

###
### Find for an initial leap-frog step-size via heuristic search.
###

function find_good_eps(
    rng::AbstractRNG,
    h::Hamiltonian,
    θ::AbstractVector{T};
    max_n_iters::Int=100
) where {T<:Real}
    ϵ′ = ϵ = 0.1
    a_min, a_cross, a_max = 0.25, 0.5, 0.75 # minimal, crossing, maximal accept ratio
    d = 2.0

    r = rand(rng, h.metric)
    z = phasepoint(h, θ, r)
    H = -neg_energy(z)

    z′ = step(Leapfrog(ϵ), h, z)
    H_new = -neg_energy(z′)

    ΔH = H - H_new
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        z′ = step(Leapfrog(ϵ′), h, z)
        H_new = -neg_energy(z′)

        ΔH = H - H_new
        DEBUG && @debug "Crossing step" direction H_new ϵ "α = $(min(1, exp(ΔH)))"
        if (direction == 1) && !(ΔH > log(a_cross))
            break
        elseif (direction == -1) && !(ΔH < log(a_cross))
            break
        else
            ϵ = ϵ′
        end
    end

    # Bisection step: ensure final accept ratio: a_min < a < a_max.
    # See https://en.wikipedia.org/wiki/Bisection_method
    ϵ, ϵ′ = ϵ < ϵ′ ? (ϵ, ϵ′) : (ϵ′, ϵ)  # ensure ϵ < ϵ′
    for _ = 1:max_n_iters
        ϵ_mid = middle(ϵ, ϵ′)
        z′ = step(Leapfrog(ϵ_mid), h, z)
        H_new = -neg_energy(z′)

        ΔH = H - H_new
        DEBUG && @debug "Bisection step" H_new ϵ_mid "α = $(min(1, exp(ΔH)))"
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


function mh_accept(
    rng::AbstractRNG,
    H::T,
    H_new::T
) where {T<:AbstractFloat}
    α = min(1.0, exp(H - H_new))
    accept = rand(rng) < α
    return accept, α
end

mh_accept(
    H::T,
    H_new::T
) where {T<:AbstractFloat} = mh_accept(GLOBAL_RNG, H, H_new)

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
