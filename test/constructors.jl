using AdvancedHMC, AbstractMCMC, Random
include("common.jl")

# Initalize samplers
nuts = NUTS(0.8)
nuts_32 = NUTS(0.8f0)
hmc = HMC(0.1, 25)
hmcda = HMCDA(0.8, 1.0)
hmcda_32 = HMCDA(0.8f0, 1.0)

integrator = Leapfrog(1e-3)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
metric = DiagEuclideanMetric(2)
adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
custom = HMCSampler(kernel, metric, adaptor)

# Check that everything is initalized correctly
@testset "Constructors" begin
    # Types
    @test typeof(nuts) == NUTS{Float64}
    @test typeof(nuts_32) == NUTS{Float32}
    @test typeof(hmc) == HMC{Float64}
    @test typeof(hmcda) == HMCDA{Float64}
    @test typeof(nuts) <: AdvancedHMC.AbstractHMCSampler
    @test typeof(nuts) <: AbstractMCMC.AbstractSampler

    # NUTS
    @test nuts.δ == 0.8
    @test nuts.max_depth == 10
    @test nuts.Δ_max == 1000.0
    @test nuts.init_ϵ == 0.0
    @test nuts.integrator == :leapfrog
    @test nuts.metric == :diagonal

    # NUTS Float32
    @test nuts_32.δ == 0.8f0
    @test nuts_32.max_depth == 10
    @test nuts_32.Δ_max == 1000.0f0
    @test nuts_32.init_ϵ == 0.0f0

    # HMC
    @test hmc.n_leapfrog == 25
    @test hmc.init_ϵ == 0.1
    @test hmc.integrator == :leapfrog
    @test hmc.metric == :diagonal

    # HMCDA
    @test hmcda.δ == 0.8
    @test hmcda.λ == 1.0
    @test hmcda.init_ϵ == 0.0
    @test hmcda.integrator == :leapfrog
    @test hmcda.metric == :diagonal

    # HMCDA Float32
    @test hmcda_32.δ == 0.8f0
    @test hmcda_32.λ == 1.0f0
    @test hmcda_32.init_ϵ == 0.0f0
end

@testset "First step" begin
    rng = MersenneTwister(0)
    θ_init = randn(rng, 2)
    logdensitymodel = AbstractMCMC.LogDensityModel(ℓπ_gdemo)
    _, nuts_state = AbstractMCMC.step(rng, logdensitymodel, nuts; n_adapts = 0, init_params = θ_init)
    _, hmc_state = AbstractMCMC.step(rng, logdensitymodel, hmc; n_adapts = 0, init_params = θ_init)
    _, nuts_32_state =
        AbstractMCMC.step(rng, logdensitymodel, nuts_32; n_adapts = 0, init_params = θ_init)
    _, custom_state = AbstractMCMC.step(rng, logdensitymodel, custom; n_adapts = 0, init_params = θ_init)

    # Metric
    @test typeof(nuts_state.metric) == DiagEuclideanMetric{Float64,Vector{Float64}}
    @test typeof(nuts_32_state.metric) == DiagEuclideanMetric{Float32,Vector{Float32}}
    @test custom_state.metric == metric

    # Integrator
    @test typeof(nuts_state.κ.τ.integrator) == Leapfrog{Float64}
    @test typeof(nuts_32_state.κ.τ.integrator) == Leapfrog{Float32}
    @test custom_state.κ.τ.integrator == integrator

    # Kernel
    @test nuts_state.κ == AdvancedHMC.make_kernel(nuts, nuts_state.κ.τ.integrator)
    @test custom_state.κ == kernel

    # Adaptor
    @test typeof(nuts_state.adaptor) <: StanHMCAdaptor
    @test hmc_state.adaptor == NoAdaptation()
    @test custom_state.adaptor == adaptor
end
