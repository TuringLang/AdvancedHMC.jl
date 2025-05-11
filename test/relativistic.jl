using ReTest, Random, AdvancedHMC

@testset "Relativistic kinetic construction" begin
    f = x -> dot(x, x)
    g = x -> 2x
    metric = UnitEuclideanMetric(10)
    h = Hamiltonian(metric, RelativisticKinetic(1.0, 1.0), f, g)
    @test h.kinetic isa RelativisticKinetic
end

@testset "Sampling with relativistic kinetic" begin
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(D)
    metric = DiagEuclideanMetric(D)
    for kineticT in [RelativisticKinetic, DimensionwiseRelativisticKinetic]
        kinetic = kineticT(1.0, 1.0)
        h = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)
        initial_ϵ = find_good_stepsize(h, initial_θ)
        integrator = Leapfrog(initial_ϵ)
        kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))
        samples, stats = sample(rng, h, kernel, initial_θ, n_samples; progress=true)
        @test length(samples) == n_samples
        @test length(stats) == n_samples
    end
end
