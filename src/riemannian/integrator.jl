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
Base.show(io::IO, l::GeneralizedLeapfrog) =
    print(io, "GeneralizedLeapfrog(ϵ=$(round.(l.ϵ; sigdigits=3)), n=$(l.n))")

# fallback to ignore return_cache & cache kwargs for other ∂H∂θ
function ∂H∂θ_cache(h, θ, r; return_cache = false, cache = nothing)
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
    n_steps::Int = 1;
    fwd::Bool = n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false),
) where {T<:AbstractScalarOrVec{<:AbstractFloat},P<:PhasePoint,FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? step_size(lf) : -step_size(lf)
    ϵ = ϵ'

    res = if FullTraj
        Vector{P}(undef, n_steps)
    else
        z
    end

    for i = 1:n_steps
        θ_init, r_init = z.θ, z.r
        # Tempering
        #r = temper(lf, r, (i=i, is_half=true), n_steps)
        # eq (16) of Girolami & Calderhead (2011)
        r_half = r_init
        local cache
        for j = 1:lf.n
            # Reuse cache for the first iteration
            if j == 1
                @unpack value, gradient = z.ℓπ
            elseif j == 2 # cache intermediate values that depends on θ only (which are unchanged)
                retval, cache = ∂H∂θ_cache(h, θ_init, r_half; return_cache = true)
                @unpack value, gradient = retval
            else # reuse cache
                @unpack value, gradient = ∂H∂θ_cache(h, θ_init, r_half; cache = cache)
            end
            r_half = r_init - ϵ / 2 * gradient
        end
        # eq (17) of Girolami & Calderhead (2011)
        θ_full = θ_init
        term_1 = ∂H∂r(h, θ_init, r_half) # unchanged across the loop
        for j = 1:lf.n
            θ_full = θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))
        end
        # eq (18) of Girolami & Calderhead (2011)
        @unpack value, gradient = ∂H∂θ(h, θ_full, r_half)
        r_full = r_half - ϵ / 2 * gradient
        # Tempering
        #r = temper(lf, r, (i=i, is_half=false), n_steps)
        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(h, θ_full, r_full; ℓπ = DualValue(value, gradient))
        # Update result
        if FullTraj
            res[i] = z
        else
            res = z
        end
        if !isfinite(z)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
    end
    return res
end
