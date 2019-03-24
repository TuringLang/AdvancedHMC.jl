abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ   ::  T
end

# Create a `Leapfrog` with a new `ϵ`
function (::Leapfrog)(ϵ::AbstractFloat)
    return Leapfrog(ϵ)
end

function is_valid(v::AbstractVector{<:Real})
    if any(isnan, v) || any(isinf, v)
        return false
    end
    return true
end

function lf_momentum(ϵ::T, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    _∂H∂θ = ∂H∂θ(h, θ)
    !is_valid(_∂H∂θ) && return r, false
    return r - ϵ * _∂H∂θ, true
end

function lf_position(ϵ::T, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
    return θ + ϵ * ∂H∂r(h, r)
end

function step(lf::Leapfrog{F}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}) where {F<:AbstractFloat,T<:Real}
    r_new, _is_valid = lf_momentum(lf.ϵ / 2, h, θ, r)
    !_is_valid && return θ, r, false
    θ_new = lf_position(lf.ϵ, h, θ, r_new)
    r_new, _is_valid = lf_momentum(lf.ϵ / 2, h, θ_new, r_new)
    !_is_valid && return θ, r, false
    return θ_new, r_new, true
end

# TODO: double check the function below to see if it is type stable or not
function steps(lf::Leapfrog{F}, h::Hamiltonian, θ::AbstractVector{T}, r::AbstractVector{T}, n_steps::Int) where {F<:AbstractFloat,T<:Real}
    n_valid = 0
    r_new, _is_valid = lf_momentum(lf.ϵ / 2, h, θ, r)
    !_is_valid && return θ, r, n_valid
    r = r_new
    for i = 1:n_steps
        θ_new = lf_position(lf.ϵ, h, θ, r)
        r_new, _is_valid = lf_momentum(i == n_steps ? lf.ϵ / 2 : lf.ϵ, h, θ, r)
        if !_is_valid
            # The reverse function below is guarantee to be numerical safe.
            # This is because we know the previous step was valid.
            r, _ = lf_momentum(-lf.ϵ / 2, h, θ, r)
            return θ, r, n_valid
        end
        θ, r = θ_new, r_new
        n_valid = n_valid + 1
    end
    return θ, r, n_valid
end
