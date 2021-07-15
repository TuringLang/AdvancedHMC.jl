using Test, Random, AdvancedHMC, ForwardDiff, AbstractMCMC
using Statistics: mean
include("common.jl")

@testset "`gdemo`" begin
    rng = MersenneTwister(0)

    n_samples = 5_000
    n_adapts = 5_000

    θ_init = randn(rng, 2)

    model = AdvancedHMC.DifferentiableDensityModel(ℓπ_gdemo, ForwardDiff)
    init_eps = Leapfrog(1e-3)
    κ = NUTS(init_eps)
    metric = DiagEuclideanMetric(2)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, κ.τ.integrator))

    samples = AbstractMCMC.sample(
        model, κ, metric, adaptor, n_adapts + n_samples;
        nadapts = n_adapts,
        init_params = θ_init,
        progress=false,
        verbose=false
    );

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est = mean(samples[n_adapts + 1:end]) do t
        t.z.θ
    end

    @test m_est ≈ [49 / 24, 7 / 6] atol=RNDATOL
end
