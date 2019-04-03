abstract type AbstractProposal end
abstract type AbstractHamiltonianTrajectory{I<:AbstractIntegrator} <: AbstractProposal end

abstract type StaticTrajectory{I<:AbstractIntegrator} <: AbstractHamiltonianTrajectory{I} end
struct TakeLastProposal{I<:AbstractIntegrator} <: StaticTrajectory{I}
    integrator  ::  I
    n_steps     ::  Int
end

# Create a `TakeLastProposal` with a new integrator
function (tlp::TakeLastProposal)(integrator::AbstractIntegrator)
    return TakeLastProposal(integrator, tlp.n_steps)
end

function transition(prop::TakeLastProposal, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    θ, r, _ = steps(prop.integrator, h, θ, r, prop.n_steps)
    return θ, -r
end

abstract type DynamicTrajectory{I<:AbstractIntegrator} <: AbstractHamiltonianTrajectory{I} end
abstract type NoUTurnTrajectory{I<:AbstractIntegrator} <: DynamicTrajectory{I} end
struct NUTS{I<:AbstractIntegrator} <: NoUTurnTrajectory{I}
    integrator  ::  I
    max_depth   ::  Int
    Δ_max       ::  AbstractFloat
end

# Helper function to use default values
NUTS(integrator::AbstractIntegrator) = NUTS(integrator, 10, 1000.0)

# Create a `NUTS` with a new integrator
function (snuts::NUTS)(integrator::AbstractIntegrator)
    return NUTS(integrator, snuts.max_depth, snuts.Δ_max)
end

struct MultinomialNUTS{I<:AbstractIntegrator} <: NoUTurnTrajectory{I}
    integrator  ::  I
end

function find_good_eps(rng::AbstractRNG, h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=100) where {T<:Real}
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

find_good_eps(h::Hamiltonian, θ::AbstractVector{T}; max_n_iters::Int=100) where {T<:Real} = find_good_eps(GLOBAL_RNG, h, θ; max_n_iters=max_n_iters)

# TODO: implement a more efficient way to build the balance tree
function build_tree(rng::AbstractRNG, nt::NoUTurnTrajectory{I}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T},
                    logu::AbstractFloat, v::Int, j::Int, H::AbstractFloat) where {I<:AbstractIntegrator,T<:Real}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        θ′, r′, _is_valid = step(nt.integrator, h, θ, r)
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

build_tree(nt::NoUTurnTrajectory{I}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T},
           logu::AbstractFloat, v::Int, j::Int, H::AbstractFloat) where {I<:AbstractIntegrator,T<:Real} = build_tree(GLOBAL_RNG, nt, h, θ, r, logu, v, j, H)

function transition(rng::AbstractRNG, nt::NoUTurnTrajectory{I}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {I<:AbstractIntegrator,T<:Real}
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

    return θ_new, r_new, α / nα
end

transition(nt::NoUTurnTrajectory{I}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {I<:AbstractIntegrator,T<:Real} = transition(GLOBAL_RNG, nt, h, θ, r)

function MultinomialNUTS(h::Hamiltonian, θ::AbstractVector{T}) where {T<:Real}
    return MultinomialNUTS(Leapfrog(find_good_eps(h, θ)))
end
