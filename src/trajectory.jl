abstract type AbstractTrajectory end

struct StaticTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory
    integrator  ::  I
    n_steps     ::  Integer
end

function points(st::StaticTrajectory, h::Hamiltonian, θ::T, r::T) where {T<:AbstractVector{<:Real}}
    ps = Vector{Tuple{T,T}}(undef, st.n_steps + 1)
    ps[1] = (θ, r)
    for i = 2:st.n_steps+1
        ps[i] = step(st.integrator, h, ps[i]...)
    end
    return ps
end

function lastpoint(st::StaticTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return steps(st.integrator, h, θ, r, st.n_steps)
end

# NOTE: the NUTS implementation is not ready
struct NoUTurnTrajectory{I<:AbstractIntegrator} <: AbstractTrajectory
    integrator  ::  I
end

# TODO: implement a more efficient way to build the balance tree
function build_tree(nt::NoUTurnTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}, logu::AbstractFloat, v::Integer, j::Integer;
                    Δ_max::AbstractFloat=1000.0) where {T<:Real}
    if j == 0
        _H = H(h, θ, r)
        # Base case - take one leapfrog step in the direction v.
        θ′, r′, _is_valid = step(nt.integrator, h, θ, r)
        # TODO: pass old H to save computation
        _H′ = _is_valid ? H(h, θ′, r′) : Inf
        n′ = (logu <= -_H′) ? 1 : 0
        s′ = (logu < Δ_max + -_H′) ? 1 : 0
        α′ = exp(min(0, _H′ -_H))

        return θ′, r′, θ′, r′, θ′, r′, n′, s′, α′, 1
    else
        # Recursion - build the left and right subtrees.
        θm, rm, θp, rp, θ′, r′, n′, s′, α′, n′α = build_tree(nt, h, θ, r, logu, v, j - 1)

        if s′ == 1
            if v == -1
                θm, rm, _, _, θ′′, r′′, n′′, s′′, α′′, n′′α = build_tree(nt, h, θm, rm, logu, v, j - 1)
            else
                _, _, θp, rp, θ′′, r′′, n′′, s′′, α′′, n′′α = build_tree(nt, h, θp, rp, logu, v, j - 1)
            end
            if rand() < n′′ / (n′ + n′′)
                θ′ = θ′′
                r′ = r′′
            end
            α′ = α′ + α′′
            n′α = n′α + n′′α
            s′ = s′′ * (dot(θp - θm, dHdr(h, rm)) >= 0 ? 1 : 0) * (dot(θp - θm, dHdr(h, rp)) >= 0 ? 1 : 0)
            n′ = n′ + n′′
        end

        return θm, rm, θp, rp, θ′, r′, n′, s′, α′, n′α
    end
end

function lastpoint(nt::NoUTurnTrajectory, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}; j_max::Integer=10) where {T<:Real}
    _H = H(h, θ, r)
    logu = log(rand()) - _H

    θm = θ; θp = θ; rm = r; rp = r; j = 0; θ_new = θ; r_new = r; n = 1; s = 1

    # local da_stats

    while s == 1 && j <= j_max

        v = rand([-1, 1])
        if v == -1
            θm, rm, _, _, θ′, r′,n′, s′, α, nα = build_tree(nt, h, θm, rm, logu, v, j)
        else
            _, _, θp, rp, θ′, r′,n′, s′, α, nα = build_tree(nt, h, θp, rp, logu, v, j)
        end

        if s′ == 1
            if rand() < min(1, n′ / n)
                θ_new = θ′; r_new = r′
            end
        end

        n = n + n′
        s = s′ * (dot(θp - θm, dHdr(h, rm)) >= 0 ? 1 : 0) * (dot(θp - θm, dHdr(h, rp)) >= 0 ? 1 : 0)
        j = j + 1

        # da_stats = α / nα

    end

    return θ_new, r_new, 2^j
end
