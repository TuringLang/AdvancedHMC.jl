# Allow pass --progress when running this script individually to turn on progress meter
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using Test, AdvancedHMC, LinearAlgebra, Random, MCMCDebugging, Plots
using Parameters: reconstruct
using Statistics: mean, var, cov
unicodeplots()
include("common.jl")

θ_init = rand(MersenneTwister(1), D)
ϵ = 0.1
n_steps = 10
n_samples = 22_000
n_adapts = 4_000

function test_stats(::Union{StaticTrajectory,HMCDA}, stats, n_adapts)
    for name in (:step_size, :nom_step_size, :n_steps, :is_accept, :acceptance_rate, :log_density, :hamiltonian_energy, :hamiltonian_energy_error, :is_adapt)
        @test all(map(s -> in(name, propertynames(s)), stats))
    end
    is_adapts = getproperty.(stats, :is_adapt)
    @test is_adapts[1:n_adapts] == ones(Bool, n_adapts)
    @test is_adapts[(n_adapts+1):end] == zeros(Bool, length(stats) - n_adapts)
end

function test_stats(::NUTS, stats, n_adapts)
    for name in (:step_size, :nom_step_size, :n_steps, :is_accept, :acceptance_rate, :log_density, :hamiltonian_energy, :hamiltonian_energy_error, :is_adapt, :max_hamiltonian_energy_error, :tree_depth, :numerical_error)
        @test all(map(s -> in(name, propertynames(s)), stats))
    end
    is_adapts = getproperty.(stats, :is_adapt)
    @test is_adapts[1:n_adapts] == ones(Bool, n_adapts)
    @test is_adapts[(n_adapts+1):end] == zeros(Bool, length(stats) - n_adapts)
end

@testset "HMC and NUTS" begin
    @testset "$metricsym" for (metricsym, metric) in Dict(
        :UnitEuclideanMetric => UnitEuclideanMetric(D),
        :DiagEuclideanMetric => DiagEuclideanMetric(D),
        :DenseEuclideanMetric => DenseEuclideanMetric(D),
    )
        @test show(metric) == nothing
        h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        @testset "$lfsym" for (lfsym, lf) in Dict(
            :Leapfrog => Leapfrog(ϵ),
            :JitteredLeapfrog => JitteredLeapfrog(ϵ, 1.0),
            :TemperedLeapfrog => TemperedLeapfrog(ϵ, 1.05),
        )
            @testset "$τsym" for (τsym, τ) in Dict(
                :(StaticTrajectory{EndPointTS}) => StaticTrajectory{EndPointTS}(lf, n_steps),
                :(StaticTrajectory{MultinomialTS}) => StaticTrajectory{MultinomialTS}(lf, n_steps),
                :(HMCDA{EndPointTS}) => HMCDA{EndPointTS}(lf, ϵ * n_steps),
                :(HMCDA{MultinomialTS}) => HMCDA{MultinomialTS}(lf, ϵ * n_steps),
                :(NUTS{SliceTS,Original}) => NUTS{SliceTS,ClassicNoUTurn}(lf),
                :(NUTS{SliceTS,Generalised}) => NUTS{SliceTS,GeneralisedNoUTurn}(lf),
                :(NUTS{MultinomialTS,Original}) => NUTS{MultinomialTS,ClassicNoUTurn}(lf),
                :(NUTS{MultinomialTS,Generalised}) => NUTS{MultinomialTS,GeneralisedNoUTurn}(lf),
            )
                @test show(h) == nothing
                @test show(τ) == nothing
                @testset  "NoAdaptation" begin
                    Random.seed!(1)
                    samples, stats = sample(h, τ, θ_init, n_samples; verbose=false, progress=PROGRESS)
                    @test mean(samples[n_adapts+1:end]) ≈ zeros(D) atol=RNDATOL
                    if "GEWEKE_TEST" in keys(ENV) && ENV["GEWEKE_TEST"] == "1"
                        res = perform(GewekeTest(5_000), mvntest, x -> rand_θ_given(x, mvntest, metric, τ), g; progress=false)
                        p = plot(res, mvntest)
                        display(p)
                        println()
                    end
                end

                # Skip adaptation tests with tempering
                if lf isa TemperedLeapfrog
                    @info "Adaptation tests for $τsym with $lfsym on $metricsym are skipped"
                    continue
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
                    @test show(adaptor) == nothing
                    Random.seed!(1)
                    # For `MassMatrixAdaptor`, we use the pre-defined step size as the method cannot adapt the step size.
                    # For other adapatation methods that are able to adpat the step size, we use `find_good_stepsize`.
                    τ_used = adaptorsym == :MassMatrixAdaptorOnly ? τ : reconstruct(τ, integrator=reconstruct(lf, ϵ=find_good_stepsize(h, θ_init)))
                    samples, stats = sample(h, τ_used , θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=PROGRESS)
                    @test mean(samples[(n_adapts+1):end]) ≈ zeros(D) atol=RNDATOL
                    test_stats(τ_used, stats, n_adapts)
                end
            end
        end
    end
end

@testset "drop_warmup" begin
    metric = DiagEuclideanMetric(D)
    h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    τ = NUTS(Leapfrog(ϵ))
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric),
        StepSizeAdaptor(0.8, τ.integrator),
    )
    samples, stats = sample(h, τ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=false, drop_warmup=true)
    @test length(samples) == n_samples - n_adapts
    @test length(stats) == n_samples - n_adapts
    samples, stats = sample(h, τ, θ_init, n_samples, adaptor, n_adapts; verbose=false, progress=false, drop_warmup=false)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end
