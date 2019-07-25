using Test, Random, AdvancedHMC
using Statistics: mean
include("common.jl")

@testset "`gdemo`" begin
    rng = MersenneTwister(0)

    n_samples = 5_000
    n_adapts = 1_000

    θ_init = randn(rng, 2)

    metric = DiagEuclideanMetric(2)
    h = Hamiltonian(metric, ℓπ_gdemo, ∂ℓπ∂θ_gdemo)
    init_eps = Leapfrog(0.1)
    prop = NUTS(init_eps)
    adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))

    samples, _ = sample(rng, h, prop, θ_init, n_samples, adaptor, n_adapts)

    m_est = mean(map(_s -> [exp(_s[1]), _s[2]], samples[1000:end]))

    @test m_est ≈ [49 / 24, 7 / 6] atol=RNDATOL
end