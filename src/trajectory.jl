####
#### Hamiltonian dynamics numerical simulation trajectories
####

abstract type AbstractProposal end
abstract type AbstractTrajectory{I<:AbstractIntegrator} <: AbstractProposal end

# Create a callback function for all `AbstractTrajectory` without passing random number generator
transition(at::AbstractTrajectory{I},
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {I<:AbstractIntegrator,T<:Real} = transition(GLOBAL_RNG, at, h, θ, r)

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
function (tlp::StaticTrajectory)(integrator::AbstractIntegrator)
    return StaticTrajectory(integrator, tlp.n_steps)
end

function transition(
    rng::AbstractRNG,
    prop::StaticTrajectory,
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {T<:Real}
    H = hamiltonian_energy(h, θ, r)
    θ_new, r_new, _ = step(prop.integrator, h, θ, r, prop.n_steps)
    H_new = hamiltonian_energy(h, θ_new, r_new)
    # Accept via MH criteria
    is_accept, α = mh_accept(rng, H, H_new)
    if is_accept
        θ, r, H = θ_new, -r_new, H_new
    end
    return θ, r, α, H
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
function (tlp::HMCDA)(integrator::AbstractIntegrator)
    return HMCDA(integrator, tlp.λ)
end

function transition(
    rng::AbstractRNG,
    prop::HMCDA,
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {T<:Real}
    # Create the corresponding static prop
    n_steps = max(1, round(Int, prop.λ / prop.integrator.ϵ))
    static_prop = StaticTrajectory(prop.integrator, n_steps)
    return transition(rng, static_prop, h, θ, r)
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

"""
Helper function to use default values
"""
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
    θ::AbstractVector{T},
    r::AbstractVector{T},
    logu::AbstractFloat,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,T<:Real}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        θ′, r′, _is_valid = step(nt.integrator, h, θ, r, v)
        H′ = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf
        n′ = (logu <= -H′) ? 1 : 0
        s′ = (logu < nt.Δ_max + -H′) ? 1 : 0
        α′ = exp(min(0, H - H′))

        return θ′, r′, θ′, r′, θ′, r′, n′, s′, α′, 1
    else
        # Recursion - build the left and right subtrees.
        θm, rm, θp, rp, θ′, r′, n′, s′, α′, n′α = build_tree(rng, nt, h, θ, r, logu, v, j - 1, H)

        if s′ == 1
            if v == -1
                θm, rm, _, _, θ′′, r′′, n′′, s′′, α′′, n′′α = build_tree(rng, nt, h, θm, rm, logu, v, j - 1, H)
            else
                _, _, θp, rp, θ′′, r′′, n′′, s′′, α′′, n′′α = build_tree(rng, nt, h, θp, rp, logu, v, j - 1, H)
            end
            if rand(rng) < n′′ / (n′ + n′′)
                θ′ = θ′′
                r′ = r′′
            end
            α′ = α′ + α′′
            n′α = n′α + n′′α
            s′ = s′′ * (dot(θp - θm, ∂H∂r(h, rm)) >= 0 ? 1 : 0) * (dot(θp - θm, ∂H∂r(h, rp)) >= 0 ? 1 : 0)
            n′ = n′ + n′′
        end

        return θm, rm, θp, rp, θ′, r′, n′, s′, α′, n′α
    end
end

build_tree(
    nt::DynamicTrajectory{I},
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T},
    logu::AbstractFloat,
    v::Int,
    j::Int,
    H::AbstractFloat
) where {I<:AbstractIntegrator,T<:Real} = build_tree(GLOBAL_RNG, nt, h, θ, r, logu, v, j, H)

function transition(
    rng::AbstractRNG,
    nt::DynamicTrajectory{I},
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {I<:AbstractIntegrator,T<:Real}
    H = hamiltonian_energy(h, θ, r)
    logu = log(rand(rng)) - H

    θm = θ; θp = θ; rm = r; rp = r; j = 0; θ_new = θ; r_new = r; n = 1; s = 1

    local α, nα
    while s == 1 && j <= nt.max_depth
        v = rand(rng, [-1, 1])
        if v == -1
            θm, rm, _, _, θ′, r′,n′, s′, α, nα = build_tree(rng, nt, h, θm, rm, logu, v, j, H)
        else
            _, _, θp, rp, θ′, r′,n′, s′, α, nα = build_tree(rng, nt, h, θp, rp, logu, v, j, H)
        end

        if s′ == 1
            if rand(rng) < min(1, n′ / n)
                θ_new = θ′
                r_new = r′
            end
        end

        n = n + n′
        s = s′ * (dot(θp - θm, ∂H∂r(h, rm)) >= 0 ? 1 : 0) * (dot(θp - θm, ∂H∂r(h, rp)) >= 0 ? 1 : 0)
        j = j + 1
    end

    H_new = 0 # Warning: NUTS always return H_new = 0;
    return θ_new, r_new, α / nα, H_new
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

    r = rand_momentum(rng, h)
    H = hamiltonian_energy(h, θ, r)

    θ′, r′, _is_valid = step(Leapfrog(ϵ), h, θ, r)
    H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

    ΔH = H - H_new
    direction = ΔH > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_n_iters
        ϵ′ = direction == 1 ? d * ϵ : 1 / d * ϵ
        θ′, r′, _is_valid = step(Leapfrog(ϵ′), h, θ′, r′)
        H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

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
        θ′, r′, _is_valid = step(Leapfrog(ϵ_mid), h, θ, r)
        H_new = _is_valid ? hamiltonian_energy(h, θ′, r′) : Inf

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


function mh_accept(rng::AbstractRNG, H::AbstractFloat, H_new::AbstractFloat)
    logα = min(0, H - H_new)
    return log(rand(rng)) < logα, exp(logα)
end
mh_accept(H::AbstractFloat, H_new::AbstractFloat) = mh_accept(GLOBAL_RNG, H, H_new)
