using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC
using Statistics: mean
include("common.jl")

@testset "AbstractMCMC w/ gdemo" begin
    rng = MersenneTwister(0)
    n_samples = 5_000
    n_adapts = 5_000
    θ_init = randn(rng, 2)
    nuts = NUTS(n_adapts = n_adapts, δ = 0.8)

    model = AdvancedHMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo),
    )

    integrator = Leapfrog(1e-3)
    κ = AdvancedHMC.make_kernel(nuts, integrator)
    metric = DiagEuclideanMetric(2)
    adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
    sampler = HMCSampler(kernel = κ, metric = metric, adaptor = adaptor)

    samples = AbstractMCMC.sample(
        rng,
        model,
        sampler,
        n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est = mean(samples[n_adapts+1:end]) do t
        t.z.θ
    end

    @test m_est ≈ [49 / 24, 7 / 6] atol = RNDATOL

    # Test that using the same AbstractRNG results in the same chain
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)
    samples1 = AbstractMCMC.sample(
        rng1,
        model,
        sampler,
        10;
        nadapts = 0,
        progress = false,
        verbose = false,
    )
    samples2 = AbstractMCMC.sample(
        rng2,
        model,
        κ,
        metric,
        adaptor,
        10;
        nadapts = 0,
        progress = false,
        verbose = false,
    )
    @test mapreduce(*, samples1, samples2) do s1, s2
        s1.z.θ == s2.z.θ
    end # Equivalent to using all, check that all samples are equal
end
