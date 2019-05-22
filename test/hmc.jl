# Allow pass --progress when running this script individually to turn on progress meter
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

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
        @testset "$(typeof(τ))" for τ in [
            StaticTrajectory(Leapfrog(ϵ), n_steps),
            HMCDA(Leapfrog(ϵ), ϵ * n_steps),
            NUTS(Leapfrog(find_good_eps(h, θ_init))),
        ]
            samples = sample(h, τ, θ_init, n_samples; verbose=false, progress=PROGRESS)
            @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL

            @testset "$(typeof(adaptor))" for adaptor in [
                Preconditioner(metric),
                NesterovDualAveraging(0.8, τ.integrator.ϵ),
                NaiveCompAdaptor(
                    Preconditioner(metric),
                    NesterovDualAveraging(0.8, τ.integrator.ϵ),
                ),
                StanNUTSAdaptor(
                    n_adapts,
                    Preconditioner(metric),
                    NesterovDualAveraging(0.8, τ.integrator.ϵ),
                ),
            ]
                samples = sample(h, τ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=PROGRESS)
                @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL
            end
        end
    end
end
