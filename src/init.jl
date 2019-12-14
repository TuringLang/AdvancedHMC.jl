function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

        struct DiffEqIntegrator{A}
          alg::A
          DiffEqIntegrator(alg::T=VelocityVerlet()) where T = new{T}(alg)
        end

        function step(
            lf::DiffEqIntegrator,
            h::Hamiltonian,
            z::P,
            n_steps::Int=1;
            fwd::Bool=n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
            res::Union{Vector{P}, P}=z)

            @unpack θ, r = z
            u0 = θ
            v0 = r
            function f1(v,u,p,t)
              ∂H∂r(h, θ)[2]
            end
            function f2(v,u,p,t)
              ∂H∂θ(h, θ)[2]
            end
            ϵ = fwd ? step_size(lf) : -step_size(lf)
            tspan = (0.0,sign(n_steps)*1.0) # Ignored and only used for direction
            prob = DynamicalODEProblem(f1,f2,v0,u0,tspan)
            integrator = init(prob,lf.alg,save_everystep=false,save_start=false,
                              save_end=false,dt=ϵ)

            for i in 1:abs(n_steps)
              step!(integrator)
            end

            integrator.u.x[2]
        end

    end
end
