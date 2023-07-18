using AdvancedHMC, AbstractMCMC

# Initalize samplers
nuts = NUTS(1000, 0.8)
hmc = HMC(0.1, 25)
hmcda = HMCDA(1000, 0.8, 1.0)

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
    @test nuts.integrator_method == Leapfrog
    @test nuts.metric_type == DiagEuclideanMetric

    # HMC
    @test hmc.n_leapfrog == 25
    @test hmc.init_ϵ == 0.1
    @test hmc.integrator_method == Leapfrog
    @test hmc.metric_type == DiagEuclideanMetric

    # HMCDA
    @test hmcda.n_adapts == 1000
    @test hmcda.δ == 0.8
    @test hmcda.λ == 1.0
    @test hmcda.init_ϵ == 0.0
    @test hmcda.integrator_method == Leapfrog
    @test hmcda.metric_type == DiagEuclideanMetric
end
