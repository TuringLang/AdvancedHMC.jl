function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin

        import .OrdinaryDiffEq

        struct DiffEqIntegrator{T<:AbstractScalarOrVec{<:AbstractFloat}, DiffEqSolver} <: AbstractLeapfrog{T}
            ϵ::T
            solver::DiffEqSolver
        end

        function step(
            integrator::DiffEqIntegrator,
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

            ϵ = fwd ? step_size(integrator) : -step_size(integrator)
            tspan = (0.0, sign(n_steps))
            problem = OrdinaryDiffEq.DynamicalODEProblem(f1, f2, v0, u0, tspan)
            diffeq_integrator = OrdinaryDiffEq.init(problem, integrator.solver, save_everystep=false, save_start=false, save_end=false, dt=ϵ)

            for i in 1:abs(n_steps)
                OrdinaryDiffEq.step!(diffeq_integrator)
                solution = diffeq_integrator.u.x  # (r, θ) at the proposed step
                z = phasepoint(h, solution[2], solution[1])
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

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin

        import Zygote

        function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractVector)
            res, back = Zygote.pullback(ℓπ, θ)
            return res[1], back(1)[1]
        end
        
        function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractMatrix)
            res, back = Zygote.pullback(ℓπ, θ)
            # FIXME: this can return Float64 when eltype(θ) is Float32
            return res, back(fill(1, size(θ)...))[1]
        end
        
        function Hamiltonian(metric, ℓπ)
            ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_zygote(ℓπ, θ)
            return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        end

    end
end
