using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC
using Statistics: mean
include("common.jl")

@testset "AbstractMCMC w/ gdemo" begin
    rng = MersenneTwister(0)
    n_samples = 5_000
    n_adapts = 5_000
    θ_init = randn(rng, 2)

    nuts = NUTS(n_adapts, 0.8)
    hmc = HMC(0.05, 100)
    hmcda = HMCDA(n_adapts, 0.8, 0.1)

    integrator = Leapfrog(1e-3)
    κ = AdvancedHMC.make_kernel(nuts, integrator)
    metric = DiagEuclideanMetric(2)
    adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
    custom = HMCSampler(κ = κ, metric = metric, adaptor = adaptor)

    model = AdvancedHMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo),
    )

    samples_nuts = AbstractMCMC.sample(
        rng,
        model,
        nuts,
        n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_nuts
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_nuts = mean(samples_nuts[n_adapts+1:end]) do t
        t.z.θ
    end

    @test m_est_nuts ≈ [49 / 24, 7 / 6] atol = RNDATOL

    samples_hmc = AbstractMCMC.sample(
        rng,
        model,
        hmc,
        n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )
    
    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_hmc
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_hmc = mean(samples_hmc) do t
        t.z.θ
    end

    @test m_est_hmc ≈ [49 / 24, 7 / 6] atol = RNDATOL

    samples_custom = AbstractMCMC.sample(
        rng,
        model,
        custom,
        n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_custom
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_custom = mean(samples_custom[n_adapts+1:end]) do t
        t.z.θ
    end

    @test m_est_custom ≈ [49 / 24, 7 / 6] atol = RNDATOL

    # Test that using the same AbstractRNG results in the same chain
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)
    samples1 = AbstractMCMC.sample(
        rng1,
        model,
        custom,
        10;
        nadapts = 0,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )
    samples2 = AbstractMCMC.sample(
        rng2,
        model,
        custom,
        10;
        nadapts = 0,
        init_params = θ_init,
        progress = false,
        verbose = false,
    )
    @test mapreduce(*, samples1, samples2) do s1, s2
        s1.z.θ == s2.z.θ
    end # Equivalent to using all, check that all samples are equal
end
