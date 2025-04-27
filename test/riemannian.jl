using ReTest, Random
using AdvancedHMC, ForwardDiff, AbstractMCMC
using LinearAlgebra

@testset "Multi variate Normal with Riemannian HMC" begin
    # Set the number of samples to draw and warmup iterations
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    # Define a Hamiltonian system
    metric = DenseRiemannianMetric((D,), ℓπ, initial_θ, λ)
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∇ℓπ)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)

    # Define an HMC sampler with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(
        rng, hamiltonian, kernel, initial_θ, n_samples; progress=true
    )
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

@testset "Multi variate Normal with Riemannian HMC softabs metric" begin
    # Set the number of samples to draw and warmup iterations
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)

    # Define a Hamiltonian system
    metric = DenseRiemannianMetric((D,), ℓπ, initial_θ, λSoftAbsMap(20.0))
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∇ℓπ)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)

    # Define an HMC sampler with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(
        rng, hamiltonian, kernel, initial_θ, n_samples; progress=true
    )
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end