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

    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin

        import .ForwardDiff, .ForwardDiff.DiffResults

        function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractVector)
            res = DiffResults.GradientResult(θ)
            ForwardDiff.gradient!(res, ℓπ, θ)
            return DiffResults.value(res), DiffResults.gradient(res)
        end

        # Implementation 1
        function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
            jacob = similar(θ)
            res = DiffResults.JacobianResult(similar(θ, size(θ, 2)), jacob)
            ForwardDiff.jacobian!(res, ℓπ, θ)
            jacob_full = DiffResults.jacobian(res)
            
            d, n = size(jacob)
            for i in 1:n
                jacob[:,i] = jacob_full[i,1+(i-1)*d:i*d]
            end
            return DiffResults.value(res), jacob
        end

        # Implementation 2
        # function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
        #     local densities
        #     f(x) = (densities = ℓπ(x); sum(densities))
        #     res = DiffResults.GradientResult(θ)
        #     ForwardDiff.gradient!(res, f, θ)
        #     return ForwardDiff.value.(densities), DiffResults.gradient(res)
        # end

        # Implementation 3
        # function ∂ℓπ∂θ_forwarddiff(ℓπ, θ::AbstractMatrix)
        #     v = similar(θ, size(θ, 2))
        #     g = similar(θ)
        #     for i in 1:size(θ, 2)
        #         res = GradientResult(θ[:,i])
        #         gradient!(res, ℓπ, θ[:,i])
        #         v[i] = value(res)
        #         g[:,i] = gradient(res)
        #     end
        #     return v, g
        # end

        function ForwardDiffHamiltonian(metric::AbstractMetric, ℓπ)
            ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_forwarddiff(ℓπ, θ)
            return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        end

        ADAVAILABLE[ForwardDiff] = ForwardDiffHamiltonian

    end

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin

        import .Zygote

        function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractVector)
            res, back = Zygote.pullback(ℓπ, θ)
            return res, first(back(Zygote.sensitivity(res)))
        end

        function ∂ℓπ∂θ_zygote(ℓπ, θ::AbstractMatrix)
            res, back = Zygote.pullback(ℓπ, θ)
            return res, first(back(ones(eltype(res), size(res))))
        end

        function ZygoteADHamiltonian(metric::AbstractMetric, ℓπ)
            ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_zygote(ℓπ, θ)
            return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        end

        ADAVAILABLE[Zygote] = ZygoteADHamiltonian
        
    end

end
