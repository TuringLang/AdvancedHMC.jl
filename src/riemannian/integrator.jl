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
    "Number of fixed-point iterations for implicit steps."
    n::Int
end

function Base.show(io::IO, l::GeneralizedLeapfrog)
    return print(io, "GeneralizedLeapfrog(ϵ=", round.(l.ϵ; sigdigits=3), ", n=", l.n, ")")
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

    for i in 1:n_steps
        θ_init, r_init = z.θ, z.r

        # Eq (16) of Girolami & Calderhead (2011) - implicit momentum half-step
        r_half = r_init
        local cache = nothing
        for j in 1:(lf.n)
            if j == 1
                # First iteration: use cached values from phase point
                (; value, gradient) = z.ℓπ
            else
                # Subsequent iterations: build/reuse cache for θ-dependent computations
                retval, cache = ∂H∂θ_cache(h, θ_init, r_half; cache=cache)
                (; value, gradient) = retval
            end
            r_half = r_init - ϵ / 2 * gradient
        end

        # Eq (17) of Girolami & Calderhead (2011) - implicit position step
        θ_full = θ_init
        term_1 = ∂H∂r(h, θ_init, r_half)  # unchanged across the loop
        for j in 1:(lf.n)
            θ_full = θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))
        end

        # Eq (18) of Girolami & Calderhead (2011) - explicit momentum half-step
        # Use the cached G_eval at θ_full to avoid a redundant metric_eval in phasepoint
        dv, cache = ∂H∂θ_cache(h, θ_full, r_half)
        (; value, gradient) = dv
        r_full = r_half - ϵ / 2 * gradient

        # Create a new phase point by caching the logdensity and gradient
        z = phasepoint(
            h, θ_full, r_full; ℓπ=DualValue(value, gradient), G_eval=cache.G_eval
        )

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
