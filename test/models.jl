using ReTest, Random, AdvancedHMC, ForwardDiff
using Statistics: mean

@testset "Models" begin
    @testset "gdemo" begin
        rng = MersenneTwister(0)

        n_samples = 5_000
        n_adapts = 1_000

        θ_init = randn(rng, 2)

        metric = DiagEuclideanMetric(2)
        h = Hamiltonian(metric, ℓπ_gdemo, ForwardDiff)
        integrator = Leapfrog(0.1)
        κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor =
            StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

        samples, _ = sample(
            rng,
            h,
            κ,
            θ_init,
            n_samples,
            adaptor,
            n_adapts;
            progress = false,
            verbose = false,
        )

        m_est = mean(map(invlink_gdemo, samples[1000:end]))

        @test m_est ≈ [49 / 24, 7 / 6] atol = RNDATOL
    end
end
