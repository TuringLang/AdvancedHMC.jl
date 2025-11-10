import AdvancedHMC: ∂H∂θ, ∂H∂r, DualValue, PhasePoint, phasepoint, step
import LinearAlgebra: norm
using AdvancedHMC: TYPEDEF, TYPEDFIELDS, AbstractScalarOrVec, AbstractLeapfrog, step_size

"""
$(TYPEDEF)

Generalized leapfrog integrator with fixed step size `ϵ`.

# Fields

$(TYPEDFIELDS)


## References 

1. Girolami, Mark, and Ben Calderhead. "Riemann manifold Langevin and Hamiltonian Monte Carlo methods." Journal of the Royal Statistical Society Series B: Statistical Methodology 73, no. 2 (2011): 123-214.
"""
struct GeneralizedLeapfrog{T<:AbstractScalarOrVec{<:AbstractFloat}} <: AbstractLeapfrog{T}
    "Step size."
    ϵ::T
    n::Int
end
function Base.show(io::IO, l::GeneralizedLeapfrog)
    return print(io, "GeneralizedLeapfrog(ϵ=", round.(l.ϵ; sigdigits=3), ", n=", l.n, ")")
end

# fallback to ignore return_cache & cache kwargs for other ∂H∂θ
function ∂H∂θ_cache(h, θ, r; return_cache=false, cache=nothing)
    dv = ∂H∂θ(h, θ, r)
    return return_cache ? (dv, nothing) : dv
end

# TODO(Kai) make sure vectorization works
# TODO(Kai) check if tempering is valid
# TODO(Kai) abstract out the 3 main steps and merge with `step` in `integrator.jl` 
function step(
    lf::GeneralizedLeapfrog{T},
    h::Hamiltonian,
    z::P,
    n_steps::Int=1;
    fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj}=Val(false),
) where {T<:AbstractScalarOrVec{<:AbstractFloat},TP,P<:PhasePoint{TP},FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    ϵ = ϵ'

    if !(T <: AbstractFloat) || !(TP <: AbstractVector)
        @warn "Vectorization is not tested for GeneralizedLeapfrog."
    end

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    diverged = false
    for i in 1:n_steps
        θ_init, r_init = z.θ, z.r
        # Tempering
        #r = temper(lf, r, (i=i, is_half=true), n_steps)
        # eq (16) of Girolami & Calderhead (2011)
        r_half = r_init
        local cache
        for j in 1:(lf.n)
            # Reuse cache for the first iteration
            if j == 1
                (; value, gradient) = z.ℓπ
            elseif j == 2 # cache intermediate values that depends on θ only (which are unchanged)
                retval, cache = ∂H∂θ_cache(h, θ_init, r_half; return_cache=true)
                (; value, gradient) = retval
            else # reuse cache
                (; value, gradient) = ∂H∂θ_cache(h, θ_init, r_half; cache=cache)
            end
            r_half = r_init - ϵ / 2 * gradient
        end

        retval, cache = ∂H∂θ_cache(h, θ_init, r_half; return_cache=true)
        (; value, gradient) = retval

        r_diff = r_half - (r_init - ϵ / 2 * gradient)
        if (norm(r_diff) > 1e-1) || any(isnan, r_half) || !isfinite(sum(r_half))
            diverged = true
            
            # Reset to last valid values
            r_half = r_init
            θ_full = θ_init

            # Recompute valid logprob/gradient for consistency
            retval, cache = ∂H∂θ_cache(h, θ_init, r_init; return_cache=true)
            (; value, gradient) = retval

            z = phasepoint(h, θ_init, r_init; ℓπ=DualValue(value, gradient))

            # If full trajectory, truncate so no undef elements remain
            if FullTraj
                res = res[1:(i-1)]
                push!(res, z)
            else
                res = z
            end

            break
        end
        # eq (17) of Girolami & Calderhead (2011)
        θ_full = θ_init
        term_1 = ∂H∂r(h, θ_init, r_half) # unchanged across the loop
        for j in 1:(lf.n)
            θ_full = θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))
        end
        θ_diff = norm(θ_full - (θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))))
        if !isfinite(sum(θ_full)) || θ_diff > 1e-1 || any(isnan, θ_full)
            diverged = true
            θ_full = θ_init
            r_full = r_init

            z = phasepoint(h, θ_init, r_init; ℓπ=DualValue(value, gradient))
            if FullTraj
                res = res[1:(i-1)]
                push!(res, z)
            else
                res = z
            end
            break
        end
        # eq (18) of Girolami & Calderhead (2011)
        (; value, gradient) = ∂H∂θ(h, θ_full, r_half)
        r_full = r_half - ϵ / 2 * gradient
        # Tempering
        #r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ_full, r_full; ℓπ=DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end

        if any(!isfinite, z.θ) || any(!isfinite, z.r)
            diverged = true
            z = phasepoint(h, θ_init, r_init; ℓπ=z.ℓπ)
        end

        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
    end
    return res, diverged
end
