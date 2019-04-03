using Test, AdvancedHMC, LinearAlgebra
using Statistics: mean, var, cov
include("common.jl")

θ_init = randn(D)
ϵ = 0.02
n_steps = 20
n_samples = 100_000
n_adapts = 2_000

@testset "HMC and NUTS" begin
    @testset "$(typeof(metric))" for metric in [
        UnitEuclideanMetric(D),
        DiagEuclideanMetric(D),
        DenseEuclideanMetric(D),
    ]
        h = Hamiltonian(metric, logπ, ∂logπ∂θ)
        @testset "$(typeof(prop))" for prop in [
            TakeLastProposal(Leapfrog(ϵ), n_steps),
            NUTS(Leapfrog(find_good_eps(h, θ_init))),
        ]
            samples = sample(h, prop, θ_init, n_samples; verbose=false)
            @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL
            @testset "$(typeof(adaptor))" for adaptor in [
                PreConditioner(metric),
                NesterovDualAveraging(0.8, prop.integrator.ϵ),
                NaiveCompAdaptor(
                    PreConditioner(metric),
                    NesterovDualAveraging(0.8, prop.integrator.ϵ),
                ),
                StanNUTSAdaptor(
                    n_adapts,
                    PreConditioner(metric),
                    NesterovDualAveraging(0.8, prop.integrator.ϵ),
                ),
            ]
                samples = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; verbose=false)
                @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL
            end
        end
    end
end
