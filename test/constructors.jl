using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

# Initalize samplers
nuts = NUTS(1000, 0.8)
nuts_32 = NUTS(1000, 0.8f0)
hmc = HMC(0.1, 25)
hmcda = HMCDA(1000, 0.8, 1.0)
hmcda_32 = HMCDA(1000, 0.8f0, 1.0)

integrator = Leapfrog(1e-3)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
metric = DiagEuclideanMetric(2)
adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
custom = HMCSampler(kernel = kernel, metric = metric, adaptor = adaptor)

# Check that everything is initalized correctly
@testset "Constructors" begin
    # Types
    @test typeof(nuts) == NUTS
    @test typeof(hmc) == HMC
    @test typeof(hmcda) == HMCDA
    @test typeof(nuts) <: AdvancedHMC.AbstractHMCSampler
    @test typeof(nuts) <: AbstractMCMC.AbstractSampler

    # NUTS
    @test nuts.n_adapts == 1000
    @test nuts.δ == 0.8
    @test nuts.max_depth == 10
    @test nuts.Δ_max == 1000.0
    @test nuts.init_ϵ == 0.0
    @test nuts.integrator == :leapfrog
    @test nuts.metric == :diagonal

    # NUTS Float32
    @test nuts.n_adapts == 1000
    @test nuts.δ == 0.8f0
    @test nuts.max_depth == 10
    @test nuts.Δ_max == 1000.0f0
    @test nuts.init_ϵ == 0.0f0
    @test nuts.integrator == :leapfrog
    @test nuts.metric == :diagonal

    # HMC
    @test hmc.n_leapfrog == 25
    @test hmc.init_ϵ == 0.1
    @test hmc.integrator == :leapfrog
    @test hmc.metric == :diagonal

    # HMCDA
    @test hmcda.n_adapts == 1000
    @test hmcda.δ == 0.8
    @test hmcda.λ == 1.0
    @test hmcda.init_ϵ == 0.0
    @test hmcda.integrator == :leapfrog
    @test hmcda.metric == :diagonal

    # HMCDA Float32
    @test hmcda.n_adapts == 1000
    @test hmcda.δ == 0.8f0
    @test hmcda.λ == 1.0f0
    @test hmcda.init_ϵ == 0.0f0
    @test hmcda.integrator == :leapfrog
    @test hmcda.metric == :diagonal
end

#=
@testset "First step" begin
    rng = MersenneTwister(0)
    _, nuts_state = step(rng, ℓπ_gdemo, nuts)
    _, nuts_32_state = step(rng, ℓπ_gdemo, nuts)
    _, hmc_state = step(rng, ℓπ_gdemo, hmc)
    _, hmcda_state = step(rng, ℓπ_gdemo, hmcda)
    _, hmcda_32_state = step(rng, ℓπ_gdemo, hmcda_32)

    # NUTS
    @test typeof(nuts_state.metric) == DiagEuclideanMetric
    @test nuts_state.kernel == 0.8
    @test nuts_state.adaptor == 10

    # NUTS Float32
    @test nuts_32_state.metric == 1000
    @test nuts_32_state.kernel == 0.8
    @test nuts_32_state.adaptor == 10

    # HMC
    @test hmc_state.metric == 1000
    @test hmc_state.kernel == 0.8
    @test hmc_state.adaptor == 10

    # HMCDA
    @test hmcda_state.metric == 1000
    @test hmcda_state.kernel == 0.8
    @test hmcda_state.adaptor == 10

    # HMCDA Float32
    @test hmcda_32_state.metric == 1000
    @test hmcda_32_state.kernel == 0.8
    @test hmcda_32_state.adaptor == 10
end
=#
