using ReTest
using AdvancedHMC, Distributions, ForwardDiff, ComponentArrays, AbstractMCMC
using LinearAlgebra, ADTypes

@testset "Demo" begin
    # Define the target distribution using the `LogDensityProblem` interface
    struct DemoProblem
        dim::Int
    end

    LogDensityProblems.logdensity(p::DemoProblem, θ) = logpdf(MvNormal(zeros(p.dim), I), θ)
    LogDensityProblems.dimension(p::DemoProblem) = p.dim
    LogDensityProblems.capabilities(::Type{DemoProblem}) =
        LogDensityProblems.LogDensityOrder{0}()

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
    proposal = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(
        hamiltonian,
        proposal,
        initial_θ,
        n_samples,
        adaptor,
        n_adapts;
        progress=false,
        verbose=false,
    )

    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

# -- test ComponentsArrays comparability

@testset "Demo ComponentsArrays" begin

    # target distribution parametrized by ComponentsArray
    p1 = ComponentVector(; μ=2.0, σ=1)
    struct DemoProblemComponentArrays end

    function LogDensityProblems.logdensity(::DemoProblemComponentArrays, p::ComponentArray)
        return -((1 - p.μ) / p.σ)^2
    end
    LogDensityProblems.dimension(::DemoProblemComponentArrays) = 2
    LogDensityProblems.capabilities(::Type{DemoProblemComponentArrays}) =
        LogDensityProblems.LogDensityOrder{0}()

    ℓπ = DemoProblemComponentArrays()

    @testset "Test Diagonal ComponentArray metric" begin

        # Define a Hamiltonian system
        M⁻¹ = ComponentArray(; μ=1.0, σ=1.0)
        metric = DiagEuclideanMetric(M⁻¹)

        # choose AD framework or provide a function manually
        hamiltonian = Hamiltonian(metric, ℓπ, Val(:ForwardDiff))

        # Define a leapfrog solver, with initial step size chosen heuristically
        initial_ϵ = find_good_stepsize(hamiltonian, p1)
        integrator = Leapfrog(initial_ϵ)

        # Define an HMC sampler, with the following components
        proposal = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )

        # -- run sampler
        n_samples, n_adapts = 100, 50
        samples, stats = sample(
            hamiltonian,
            proposal,
            p1,
            n_samples,
            adaptor,
            n_adapts;
            progress=false,
            verbose=false,
        )

        @test length(samples) == n_samples
        @test length(stats) == n_samples
        lab = ComponentArrays.labels(samples[1])
        @test "μ" ∈ lab
        @test "σ" ∈ lab
    end

    @testset "Test Dense ComponentArray metric" begin

        # Define a Hamiltonian system
        ax = getaxes(p1)[1]
        M⁻¹ = ComponentArray([2.0 1.0; 1.0 2.0], ax, ax)
        metric = DenseEuclideanMetric(M⁻¹)

        # choose AD framework or provide a function manually
        hamiltonian = Hamiltonian(metric, ℓπ, Val(:ForwardDiff))

        # Define a leapfrog solver, with initial step size chosen heuristically
        initial_ϵ = find_good_stepsize(hamiltonian, p1)
        integrator = Leapfrog(initial_ϵ)

        # Define an HMC sampler, with the following components
        proposal = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )

        # -- run sampler
        n_samples, n_adapts = 100, 50
        samples, stats = sample(
            hamiltonian,
            proposal,
            p1,
            n_samples,
            adaptor,
            n_adapts;
            progress=false,
            verbose=false,
        )

        @test length(samples) == n_samples
        @test length(stats) == n_samples
        lab = ComponentArrays.labels(samples[1])
        @test "μ" ∈ lab
        @test "σ" ∈ lab
    end
end

@testset "ADTypes" begin
    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 2_000, 1_000
    initial_θ = rand(D)
    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(D)

    hamiltonian_ldp = Hamiltonian(metric, ℓπ_gdemo, AutoForwardDiff())

    model = AbstractMCMC.LogDensityModel(ℓπ_gdemo)
    hamiltonian_ldm = Hamiltonian(metric, model, AutoForwardDiff())

    for hamiltonian in (hamiltonian_ldp, hamiltonian_ldm)
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)

        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )

        samples, stats = sample(
            hamiltonian,
            kernel,
            initial_θ,
            n_samples,
            adaptor,
            n_adapts;
            progress=false,
            verbose=false,
        )

        @test length(samples) == n_samples
        @test length(stats) == n_samples
    end
end
