####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ   ::  T
end

# Create a `Leapfrog` with a new `ϵ`
function (::Leapfrog)(ϵ::AbstractFloat)
    return Leapfrog(ϵ)
end

function lf_momentum(
    ϵ::T,
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {T<:Real}
    _∂H∂θ = ∂H∂θ(h, θ)
    !is_valid(_∂H∂θ) && return r, false
    return r - ϵ * _∂H∂θ, true
end

function lf_position(
    ϵ::T, h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T}
) where {T<:Real}
    return θ + ϵ * ∂H∂r(h, r)
end

# TODO: double check the function below to see if it is type stable or not
function step(
    lf::Leapfrog{F},
    h::Hamiltonian,
    θ::AbstractVector{T},
    r::AbstractVector{T},
    n_steps::Int=1
) where {F<:AbstractFloat,T<:Real}
    fwd = n_steps > 0 # simulate hamiltonian backward when n_steps < 0
    ϵ = fwd ? lf.ϵ : -lf.ϵ
    n_valid = 0

    r_new, _is_valid_1 = lf_momentum(ϵ/2, h, θ, r)
    for i = 1:abs(n_steps)
        θ_new = lf_position(ϵ, h, θ, r_new)
        r_new, _is_valid_2 = lf_momentum(i == n_steps ? ϵ / 2 : ϵ, h, θ_new, r_new)
        if (_is_valid_1 && _is_valid_2)
            θ, r = θ_new, r_new
            n_valid = n_valid + 1
        else
            # Reverse half leapfrog step for r when breaking
            #  the loop immaturely.
            if i > 1 && i < abs(n_steps)
                r, _ = lf_momentum(-lf.ϵ / 2, h, θ, r)
            end
            break
        end
    end
    return θ, r, n_valid > 0
end

###
### Utility function.
###

function is_valid(v::AbstractVector{<:Real})
    if any(isnan, v) || any(isinf, v)
        return false
    else
        return true
    end
end
