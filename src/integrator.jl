abstract type AbstractIntegrator end

struct Leapfrog{T<:Real} <: AbstractIntegrator
    ϵ   ::  T
end

function _is_valid(v::AbstractVector{<:Real})
    if any(isnan, v) || any(isinf, v)
        return false
    end
    return true
end

function _lf_momentum(ϵ::T, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    dHdθ = _dHdθ(h, θ)
    !_is_valid(dHdθ) && return r, false
    return r - ϵ .* dHdθ, true
end

function _lf_position(ϵ::T, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return θ + ϵ .* _dHdr(h, r)
end

function step(lf::Leapfrog{T}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    r_new, is_valid = _lf_momentum(lf.ϵ / 2, h, θ, r)
    !is_valid && return θ, r, false
    θ_new = _lf_position(lf.ϵ, h, θ, r_new)
    r_new, is_valid = _lf_momentum(lf.ϵ / 2, h, θ_new, r_new)
    !is_valid && return θ, r, false
    return θ_new, r_new, true
end

function steps(lf::Leapfrog{T}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}, n_steps::Integer) where {T<:Real}
    n_valid = 0
    r_new, is_valid = _lf_momentum(lf.ϵ / 2, h, θ, r)
    !is_valid && return θ, r, n_valid
    r = r_new
    for i = 1:n_steps
        θ_new = _lf_position(lf.ϵ, h, θ, r)
        r_new, is_valid = _lf_momentum(i == n_steps ? lf.ϵ / 2 : lf.ϵ, h, θ, r)
        if !is_valid
            # The reverse function below is guarantee to be numerical safe.
            # This is because we know the previous step was valid.
            r, _ = _lf_momentum(-lf.ϵ / 2, h, θ, r)
            return θ, r, n_valid
        end
        θ, r = θ_new, r_new
        n_valid = n_valid + 1
    end
    return θ, r, n_valid
end
