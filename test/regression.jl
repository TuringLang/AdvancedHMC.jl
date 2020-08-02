using Test, Random, BSON, AdvancedHMC, Distributions, ForwardDiff

function simulate_and_compare(hmc_variant)
    Random.seed!(1)

    dim = 10; θ₀ = rand(dim)

    ℓπ(θ) = logpdf(MvNormal(zeros(dim), ones(dim)), θ)

    n_samples, n_adapts = 2_000, 1_000

    metric = DiagEuclideanMetric(dim)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    ϵ₀ = find_good_stepsize(hamiltonian, θ₀)
    integrator = Leapfrog(ϵ₀)

    κ = if hmc_variant == "hmc_mh"
        HMCKernel(FullRefreshment(), Trajectory(integrator, FixedNSteps(10)), MetropolisTS)
    elseif hmc_variant == "hmc_multi"
        HMCKernel(FullRefreshment(), Trajectory(integrator, FixedNSteps(10)), MultinomialTS)
    elseif hmc_variant == "nuts_slice"
        HMCKernel(FullRefreshment(), Trajectory(integrator, NoUTurn()), SliceTS)
    elseif hmc_variant == "nuts_multi"
        HMCKernel(FullRefreshment(), Trajectory(integrator, NoUTurn()), MultinomialTS)
    end

    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, ϵ₀))

    samples, stats = sample(hamiltonian, κ, θ₀, n_samples, adaptor, n_adapts; progress=true)

    old = BSON.load("$(@__DIR__)/regression/ef6de39/$hmc_variant.bson")

    @test samples == old[:samples]
    @test stats == old[:stats]
end

@testset "Regression" begin
    for hmc_variant in ["hmc_mh", "hmc_multi", "nuts_slice", "nuts_multi"]
        simulate_and_compare(hmc_variant)
    end
end
