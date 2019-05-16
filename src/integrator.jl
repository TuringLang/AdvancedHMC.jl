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

# TODO: double check the function below to see if it is type stable or not
function step(
    lf::Leapfrog{F},
    h::Hamiltonian,
    #θ::AbstractVector{T},
    #r::AbstractVector{T},
    z::PhasePoint,
    n_steps::Int=1;
    fwd = n_steps > 0 # simulate hamiltonian backward when n_steps < 0
) where {F<:AbstractFloat,T<:Real}
    @unpack θ, r = z
    ϵ = fwd ? lf.ϵ : -lf.ϵ

    # r_new, _is_valid_1 = lf_momentum(ϵ/2, h, θ, r)
    ∇θ = ∂H∂θ(h, θ)
    for i = 1:abs(n_steps)
        r = r - ϵ/2 * ∇θ # Take a half leapfrog step for momentum variable
        ∇r = ∂H∂r(h, r)
        # θ_new = lf_position(ϵ, h, θ, r_new)
        θ = θ + ϵ * ∇r # Take a full leapfrog step for position variable
        ∇θ = ∂H∂θ(h, θ)
        # r_new, _is_valid_2 = lf_momentum(i == n_steps ? ϵ / 2 : ϵ, h, θ_new, r_new)
        r = r - ϵ/2 * ∇θ # Take a half leapfrog step for momentum variable
        # if (_is_valid_1 && _is_valid_2)
        #     θ, r = θ_new, r_new
        # else
        #     # Reverse half leapfrog step for r when breaking
        #     #  the loop immaturely.
        #     if i > 1 && i < abs(n_steps)
        #         r, _ = lf_momentum(-lf.ϵ / 2, h, θ, r)
        #     end
        #     break
        # end
        z = phasepoint(h, θ, r)
    end
    # return θ, r
    return z
end
