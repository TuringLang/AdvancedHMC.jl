using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC, MCMCChains
using Statistics: mean

@testset "MCMCChains w/ gdemo" begin
    rng = MersenneTwister(0)

    n_samples = 5_000
    n_adapts = 5_000

    θ_init = randn(rng, 2)

    model = AdvancedHMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo),
    )
    integrator = Leapfrog(1e-3)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    metric = DiagEuclideanMetric(2)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    sampler = HMCSampler(kernel, metric, adaptor)

    samples = AbstractMCMC.sample(
        rng,
        model,
        sampler,
        n_adapts + n_samples;
        nadapts = n_adapts,
        initial_params = θ_init,
        chain_type = Chains,
        progress = false,
        bijector = invlink_gdemo,
        verbose = false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    m_est = mean(samples[n_adapts+1:end])

    @test m_est[:, 2] ≈ [49 / 24, 7 / 6] atol = RNDATOL
end
