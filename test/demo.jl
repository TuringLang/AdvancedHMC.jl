using ReTest
using AdvancedHMC, Distributions, ForwardDiff, ComponentArrays
using LinearAlgebra

@testset "Demo" begin
    # Define the target distribution using the `LogDensityProblem` interface
    struct DemoProblem
        dim::Int
    end
    LogDensityProblems.logdensity(p::DemoProblem, θ) = logpdf(MvNormal(zeros(p.dim), I), θ)
    LogDensityProblems.dimension(p::DemoProblem) = p.dim

    # Choose parameter dimensionality and initial parameter value
    D = 10
    initial_θ = rand(D)
    ℓπ = DemoProblem(D)

    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 2_000, 1_000

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=false, verbose=false)

    @test length(samples) == n_samples
    @test length(stats) == n_samples

end

# -- test ComponentsArrays comparability

@testset "Demo ComponentsArrays" begin

    # target distribution parametrized by ComponentsArray
    p1 = ComponentVector(μ=2.0, σ=1)
    struct DemoProblemComponentArrays end
    function LogDensityProblems.logdensity(::DemoProblemComponentArrays, p::ComponentArray)
        return -((1 - p.μ) / p.σ)^2
    end
    LogDensityProblems.dimension(::DemoProblemComponentArrays) = 2
    ℓπ = DemoProblemComponentArrays()

    # Define a Hamiltonian system
    D = length(p1)          # number of parameters
    metric = DiagEuclideanMetric(D)

    # choose AD framework or provide a function manually
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, p1)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # -- run sampler
    n_samples, n_adapts = 100, 50
    samples, stats = sample(hamiltonian, proposal, p1, n_samples,
                            adaptor, n_adapts; progress=false, verbose=false)

    @test length(samples) == n_samples
    @test length(stats) == n_samples
    labels = ComponentArrays.labels(samples[1])
    @test "μ" ∈ labels
    @test "σ" ∈ labels

end
