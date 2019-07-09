# Allow pass --progress when running this script individually to turn on progress meter
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using Test, AdvancedHMC, LinearAlgebra
using Statistics: mean, var, cov
include("common.jl")

θ_init = 2rand(D)
ϵ = 0.02
n_steps = 20
n_samples = 100_000
n_adapts = 2_000

@testset "HMC and NUTS" begin
    @testset "$metricsym" for (metricsym, metric) in Dict(
        :UnitEuclideanMetric => UnitEuclideanMetric(D),
        :DiagEuclideanMetric => DiagEuclideanMetric(D),
        :DenseEuclideanMetric => DenseEuclideanMetric(D),
    )
        h = Hamiltonian(metric, logπ, ∂logπ∂θ)
        @testset "$τsym" for (τsym, τ) in Dict(
            :HMC => StaticTrajectory(Leapfrog(ϵ), n_steps),
            :HMCDA => HMCDA(Leapfrog(ϵ), ϵ * n_steps),
            :MultinomialNUTS => NUTS(Leapfrog(find_good_eps(h, θ_init)); sampling=:multinomial),
            :SliceNUTS => NUTS(Leapfrog(find_good_eps(h, θ_init)); sampling=:slice),
        )
            samples = sample(h, τ, θ_init, n_samples; verbose=false, progress=PROGRESS)
            @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL

            @testset "$adaptorsym" for (adaptorsym, adaptor) in Dict(
                :PreconditionerOnly => Preconditioner(metric),
                :NesterovDualAveragingOnly => NesterovDualAveraging(0.8, τ.integrator.ϵ),
                :NaiveHMCAdaptor => NaiveHMCAdaptor(
                    Preconditioner(metric),
                    NesterovDualAveraging(0.8, τ.integrator.ϵ),
                ),
                :StanHMCAdaptor => StanHMCAdaptor(
                    n_adapts,
                    Preconditioner(metric),
                    NesterovDualAveraging(0.8, τ.integrator.ϵ),
                ),
            )
                samples = sample(h, τ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=PROGRESS)
                @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL
            end
        end
    end
end
