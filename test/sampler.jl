# Allow pass --progress when running this script individually to turn on progress meter
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using ReTest, AdvancedHMC, LinearAlgebra, Random, Plots
using AdvancedHMC: StaticTerminationCriterion, DynamicTerminationCriterion
using Setfield
using Statistics: mean, var, cov
unicodeplots()
include("common.jl")

function test_stats(::Trajectory{TS,I,TC}, stats, n_adapts) where {TS,I,TC<:StaticTerminationCriterion}
    for name in (:step_size, :nom_step_size, :n_steps, :is_accept, :acceptance_rate, :log_density, :hamiltonian_energy, :hamiltonian_energy_error, :is_adapt)
        @test all(map(s -> in(name, propertynames(s)), stats))
    end
    is_adapts = getproperty.(stats, :is_adapt)
    @test is_adapts[1:n_adapts] == ones(Bool, n_adapts)
    @test is_adapts[(n_adapts+1):end] == zeros(Bool, length(stats) - n_adapts)
end

function test_stats(::Trajectory{TS,I,TC}, stats, n_adapts) where {TS,I,TC<:DynamicTerminationCriterion}
    for name in (:step_size, :nom_step_size, :n_steps, :is_accept, :acceptance_rate, :log_density, :hamiltonian_energy, :hamiltonian_energy_error, :is_adapt, :max_hamiltonian_energy_error, :tree_depth, :numerical_error)
        @test all(map(s -> in(name, propertynames(s)), stats))
    end
    is_adapts = getproperty.(stats, :is_adapt)
    @test is_adapts[1:n_adapts] == ones(Bool, n_adapts)
    @test is_adapts[(n_adapts+1):end] == zeros(Bool, length(stats) - n_adapts)
end

@testset "sample" begin
    θ_init = rand(MersenneTwister(1), D)
    ϵ = 0.1
    n_steps = 10
    n_samples = 22_000
    n_adapts = 4_000
    @testset "$metricsym" for (metricsym, metric) in Dict(
        :UnitEuclideanMetric => UnitEuclideanMetric(D),
        :DiagEuclideanMetric => DiagEuclideanMetric(D),
        :DenseEuclideanMetric => DenseEuclideanMetric(D),
    )
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        @testset "$lfsym" for (lfsym, lf) in Dict(
            :Leapfrog => Leapfrog(ϵ),
            :JitteredLeapfrog => JitteredLeapfrog(ϵ, 1.0),
            :TemperedLeapfrog => TemperedLeapfrog(ϵ, 1.05),
        )
            @testset "$τstr" for (τstr, τ) in Dict(
                "Trajectory{EndPointTS,FixedNSteps}" => Trajectory{EndPointTS}(lf, FixedNSteps(n_steps)),
                "Trajectory{MultinomialTS,FixedNSteps}" => Trajectory{MultinomialTS}(lf, FixedNSteps(n_steps)),
                "Trajectory{EndPointTS,FixedIntegrationTime}" => Trajectory{EndPointTS}(lf, FixedIntegrationTime(ϵ * n_steps)),
                "Trajectory{MultinomialTS,FixedIntegrationTime}" => Trajectory{MultinomialTS}(lf, FixedIntegrationTime(ϵ * n_steps)),
                "Trajectory{SliceTS,Original}" => Trajectory{SliceTS}(lf, ClassicNoUTurn()),
                "Trajectory{SliceTS,Generalised}" => Trajectory{SliceTS}(lf, GeneralisedNoUTurn()),
                "Trajectory{SliceTS,StrictGeneralised}" => Trajectory{SliceTS}(lf, StrictGeneralisedNoUTurn()),
                "Trajectory{MultinomialTS,Original}" => Trajectory{MultinomialTS}(lf, ClassicNoUTurn()),
                "Trajectory{MultinomialTS,Generalised}" => Trajectory{MultinomialTS}(lf, GeneralisedNoUTurn()),
                "Trajectory{MultinomialTS,StrictGeneralised}" => Trajectory{MultinomialTS}(lf, StrictGeneralisedNoUTurn()),
            )
                @testset "Base.show" begin
                    test_show(metric)
                    test_show(h)
                    test_show(τ)
                end

                @testset  "NoAdaptation" begin
                    Random.seed!(1)
                    samples, stats = sample(h, HMCKernel(τ), θ_init, n_samples; verbose=false, progress=PROGRESS)
                    @test mean(samples) ≈ zeros(D) atol=RNDATOL
                end

                @testset "$adaptorsym" for (adaptorsym, adaptor) in Dict(
                    :MassMatrixAdaptorOnly => MassMatrixAdaptor(metric),
                    :StepSizeAdaptorOnly => StepSizeAdaptor(0.8, τ.integrator),
                    :NaiveHMCAdaptor => NaiveHMCAdaptor(
                        MassMatrixAdaptor(metric),
                        StepSizeAdaptor(0.8, τ.integrator),
                    ),
                    :StanHMCAdaptor => StanHMCAdaptor(
                        MassMatrixAdaptor(metric),
                        StepSizeAdaptor(0.8, τ.integrator),
                    ),
                )
                    # Skip adaptation tests with tempering
                    if lf isa TemperedLeapfrog
                        continue
                    end

                    test_show(adaptor)

                    Random.seed!(1)
                    # For `MassMatrixAdaptor`, we use the pre-defined step size as the method cannot adapt the step size.
                    # For other adapatation methods that are able to adapt the step size, we use `find_good_stepsize`.
                    τ_used = if adaptorsym == :MassMatrixAdaptorOnly
                        τ
                    else
                        ϵ_used = find_good_stepsize(h, θ_init)
                        @set τ.integrator.ϵ = ϵ_used
                    end
                    samples, stats = sample(h, HMCKernel(τ_used) , θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=PROGRESS)
                    @test mean(samples[(n_adapts+1):end]) ≈ zeros(D) atol=RNDATOL
                    test_stats(τ_used, stats, n_adapts)
                end
            end
        end
    end
    @testset "drop_warmup" begin
        metric = DiagEuclideanMetric(D)
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        κ = NUTS(Leapfrog(ϵ))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric),
            StepSizeAdaptor(0.8, κ.τ.integrator),
        )
        samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=false, drop_warmup=true)
        @test length(samples) == n_samples - n_adapts
        @test length(stats) == n_samples - n_adapts
        samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=false, drop_warmup=false)
        @test length(samples) == n_samples
        @test length(stats) == n_samples
    end
end
