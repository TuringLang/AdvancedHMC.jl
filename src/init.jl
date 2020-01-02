function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin

        using .OrdinaryDiffEq

        struct DiffEqIntegrator{T<:AbstractScalarOrVec{<:AbstractFloat},A} <: AbstractLeapfrog{T}
            ϵ::T
            alg::A
            DiffEqIntegrator(ϵ::T, alg::A=VelocityVerlet()) where {T,A} = new{T,A}(ϵ, alg)
        end

        function step(
            lf::DiffEqIntegrator,
            h::Hamiltonian,
            z::P,
            n_steps::Int=1;
            fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
            res::Union{Vector{P}, P}=z) where {P<:PhasePoint}

            @unpack θ, r = z

            # For DynamicalODEProblem `u` is `θ` and `v` is `r`
            # f1 is dr/dt RHS function
            # f2 is dθ/dt RHS function
			v0, u0 = r, θ

			f1(v, u, p, t) = -∂H∂θ(h, u).gradient
			f2(v, u, p, t) = ∂H∂r(h, v)

            ϵ = fwd ? step_size(lf) : -step_size(lf)
            tspan = (0.0, sign(n_steps))
            prob = DynamicalODEProblem(f1, f2, v0, u0, tspan)
            integrator = init(prob, lf.alg, save_everystep=false, save_start=false, save_end=false, dt=ϵ)

            for i in 1:abs(n_steps)
                step!(integrator)
                z = phasepoint(h, integrator.u.x[2], integrator.u.x[1])
                !isfinite(z) && break
                if res isa Vector
                    res[i] = z
                else
                    res = z
                end
            end
			return res
		end

		export DiffEqIntegrator

    end
end
