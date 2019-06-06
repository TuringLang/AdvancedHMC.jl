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

###
### Standard HMC implementation with fixed leapfrog step numbers.
###
struct StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end

"""
Termination (i.e. no-U-turn).
"""
struct Termination end

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

"""
Dynamic trajectory HMC using the no-U-turn termination criteria algorithm.
"""
struct NUTS{I<:AbstractIntegrator} <: DynamicTrajectory{I}
    integrator  ::  I
    max_depth   ::  Int
    Δ_max       ::  AbstractFloat
end


# Helper function to use default values
NUTS(integrator::AbstractIntegrator) = NUTS(integrator, 10, 1000.0)

"""
Create a `NUTS` with a new integrator
"""
function (snuts::NUTS)(integrator::AbstractIntegrator)
    return NUTS(integrator, snuts.max_depth, snuts.Δ_max)
end


###
### The doubling tree algorithm for expanding trajectory.
###

# TODO: implement a more efficient way to build the balance tree
function build_tree(
    rng::AbstractRNG,
    nt::DynamicTrajectory{I},
    h::Hamiltonian,
    z::PhasePoint,
    logu::AbstractFloat,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,T<:Real}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = step(nt.integrator, h, z, v)
        H′ = -neg_energy(z′)
        n′ = (logu <= -H′) ? 1 : 0
        s′ = (logu < nt.Δ_max + -H′) ? 1 : 0
        α′ = exp(min(0, H - H′))

        return z′, z′, z′, n′, s′, α′, 1
    else
        # Recursion - build the left and right subtrees.
        zm, zp, z′, n′, s′, α′, n′α = build_tree(rng, nt, h, z, logu, v, j - 1, H)

        if s′ == 1
            if v == -1
                zm, _, z′′, n′′, s′′, α′′, n′′α = build_tree(rng, nt, h, zm, logu, v, j - 1, H)
            else
                _, zp, z′′, n′′, s′′, α′′, n′′α = build_tree(rng, nt, h, zp, logu, v, j - 1, H)
            end
            if rand(rng) < n′′ / (n′ + n′′)
                z′ = z′′
            end
            α′ = α′ + α′′
            n′α = n′α + n′′α
            s′ = s′′ * (dot(zp.θ - zm.θ, ∂H∂r(h, zm.r)) >= 0 ? 1 : 0) * (dot(zp.θ - zm.θ, ∂H∂r(h, zp.r)) >= 0 ? 1 : 0)
            n′ = n′ + n′′
        end

        # s: termination stats
        # α: MH stats, i.e. sum of MH accept prob for all leapfrog steps
        # nα: total # of leap frog steps, i.e. phase points in a trajectory
        # n: # of acceptable candicates, i.e. prob is larger than slice variable u
        return zm, zp, z′, n′, s′, α′, n′α
    end
end

build_tree(
    nt::DynamicTrajectory{I},
    h::Hamiltonian,
    z::PhasePoint,
    logu::AbstractFloat,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,T<:Real} = build_tree(GLOBAL_RNG, nt, h, z, logu, v, j, H)

function transition(
    rng::AbstractRNG,
    nt::DynamicTrajectory{I},
    h::Hamiltonian,
    z::PhasePoint
) where {I<:AbstractIntegrator,T<:Real}
    θ, r = z.θ, z.r
    H = -neg_energy(z)
    logu = log(rand(rng)) - H

    zm = z; zp = z; z_new = z; j = 0; n = 1; s = 1

    local α, nα
    while s == 1 && j <= nt.max_depth
        v = rand(rng, [-1, 1])
        if v == -1
            zm, _, z′, n′, s′, α, nα = build_tree(rng, nt, h, zm, logu, v, j, H)
        else
            zm, _, z′, n′, s′, α, nα = build_tree(rng, nt, h, zm, logu, v, j, H)
        end

        if s′ == 1
            if rand(rng) < min(1, n′ / n)
                z_new = z′
            end
        end

        n = n + n′
        s = s′ * (dot(zp.θ - zm.θ, ∂H∂r(h, zm.r)) >= 0 ? 1 : 0) * (dot(zp.θ - zm.θ, ∂H∂r(h, zp.r)) >= 0 ? 1 : 0)
        j = j + 1
    end

    return z_new, α / nα
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
