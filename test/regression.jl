using Test, Random, BSON, AdvancedHMC, Distributions, ForwardDiff

is_ef6de39 = isdefined(AdvancedHMC, :EndPointTS)

function simulate_and_compare(hmc_variant)
    filepath_ef6de39 = "$(@__DIR__)/regression/ef6de39/$hmc_variant.bson"

    Random.seed!(1)

    dim = 5; θ₀ = rand(dim)

    ℓπ(θ) = logpdf(MvNormal(zeros(dim), ones(dim)), θ)

    n_samples, n_adapts = 2_000, 1_000

    metric = DiagEuclideanMetric(dim)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    ϵ₀ = find_good_stepsize(hamiltonian, θ₀)
    integrator = Leapfrog(ϵ₀)

    if is_ef6de39
        κ = if hmc_variant == "hmc_mh"
            StaticTrajectory{EndPointTS}(integrator, 10)
        elseif hmc_variant == "hmc_multi"
            StaticTrajectory{MultinomialTS}(integrator, 10)
        elseif hmc_variant == "nuts_slice"
            NUTS{SliceTS, GeneralisedNoUTurn}(integrator)
        elseif hmc_variant == "nuts_multi"
            NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
        end
    else
        κ = if hmc_variant == "hmc_mh"
            HMCKernel(Trajectory(integrator, FixedNSteps(10)), MetropolisTS)
        elseif hmc_variant == "hmc_multi"
            HMCKernel(Trajectory(integrator, FixedNSteps(10)), MultinomialTS)
        elseif hmc_variant == "nuts_slice"
            HMCKernel(Trajectory(integrator, NoUTurn()), SliceTS)
        elseif hmc_variant == "nuts_multi"
            HMCKernel(Trajectory(integrator, NoUTurn()), MultinomialTS)
        end
    end

    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, ϵ₀))

    samples, stats = sample(
        hamiltonian, κ, θ₀, n_samples, adaptor, n_adapts; verbose=false, progress=false
    )

    is_ef6de39 && return BSON.bson(filepath_ef6de39, samples=samples, stats=stats)

    bson_ef6de39 = BSON.load(filepath_ef6de39)
    @test samples == bson_ef6de39[:samples]
    @test stats == bson_ef6de39[:stats]
end

if "REGRESSION_TEST" in keys(ENV) && ENV["REGRESSION_TEST"] == "1"
    @testset "Regression" begin
        for hmc_variant in [
            "hmc_mh", "hmc_multi", 
            # "nuts_slice", "nuts_multi"
        ]
            simulate_and_compare(hmc_variant)
        end
    end
end
