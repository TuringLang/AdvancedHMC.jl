using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD

mutable struct CountingLogDensity
    n_gradient_calls::Int
end

LogDensityProblems.dimension(::CountingLogDensity) = 2
LogDensityProblems.logdensity(::CountingLogDensity, θ) = -sum(abs2, θ) / 2
function LogDensityProblems.logdensity_and_gradient(ℓ::CountingLogDensity, θ)
    ℓ.n_gradient_calls += 1
    return LogDensityProblems.logdensity(ℓ, θ), -θ
end
function LogDensityProblems.capabilities(::Type{<:CountingLogDensity})
    return LogDensityProblems.LogDensityOrder{1}()
end

@testset "SGLD" begin
    @testset "PolynomialStepsize" begin
        p = PolynomialStepsize(0.25)
        @test p.a == 0.25
        @test p.b == 0
        @test p.γ == 0.55
        @test eltype(p) == Float64
        @test p(4) == 0.25 / 4^0.55

        promoted = PolynomialStepsize(Float32(0.25), 10.0, Float32(1))
        @test eltype(promoted) == Float64
        @test promoted(2) == 0.25 / 12

        @test_throws ErrorException PolynomialStepsize(0.25, 0, 0.5)
        @test_throws ErrorException PolynomialStepsize(0.25, 0, 1.1)
    end

    @testset "getparams and setparams!!" begin
        rng = MersenneTwister(1)
        θ_init = randn(rng, 2)
        new_θ = randn(rng, 2)
        model = AbstractMCMC.LogDensityModel(CountingLogDensity(0))
        sgld = SGLD(PolynomialStepsize(0.25); metric=:unit)

        t, s = AbstractMCMC.step(rng, model, sgld; initial_params=θ_init)
        @test AbstractMCMC.getparams(s) == t.z.θ
        @test t.z.r == zero(t.z.θ)
        new_state = AbstractMCMC.setparams!!(model, s, new_θ)
        @test AbstractMCMC.getparams(new_state) == new_θ
        @test new_state.transition.z.r == zero(new_θ)
        @test new_state.metric == s.metric
    end

    @testset "Welling-Teh update" begin
        θ = [0.5, -1.0]
        stepsize = PolynomialStepsize(0.25, 0.0, 1.0)
        ℓ = CountingLogDensity(0)
        counted_model = AbstractMCMC.LogDensityModel(ℓ)

        for metric in (
            UnitEuclideanMetric(2),
            DiagEuclideanMetric([2.0, 0.5]),
            DenseEuclideanMetric([2.0 0.3; 0.3 1.0]),
        )
            step_rng = MersenneTwister(2)
            noise_rng = MersenneTwister(2)
            ξ = randn(noise_rng, Float64, length(θ))
            ϵ = stepsize(1)
            grad = -θ
            drift = if metric isa UnitEuclideanMetric
                grad
            elseif metric isa DiagEuclideanMetric
                metric.M⁻¹ .* grad
            else
                metric.M⁻¹ * grad
            end
            noise = if metric isa UnitEuclideanMetric
                ξ
            elseif metric isa DiagEuclideanMetric
                metric.sqrtM⁻¹ .* ξ
            else
                metric.cholM⁻¹' * ξ
            end
            expected = θ .+ (ϵ / 2) .* drift .+ sqrt(ϵ) .* noise

            t, state = AbstractMCMC.step(
                step_rng, counted_model, SGLD(stepsize; metric=metric); initial_params=θ
            )

            @test t.z.θ ≈ expected
            @test t.z.r == zero(θ)
            @test t.z.ℓκ.value == 0
            @test t.stat.n_steps == 1
            @test t.stat.step_size == ϵ
            @test t.stat.is_accept == true
            @test t.stat.acceptance_rate == 1
            @test t.stat.is_adapt == false
            @test state.metric == metric
        end

        count_rng = MersenneTwister(3)
        _, state = AbstractMCMC.step(
            count_rng, counted_model, SGLD(stepsize); initial_params=θ
        )
        ℓ.n_gradient_calls = 0
        t, state = AbstractMCMC.step(count_rng, counted_model, SGLD(stepsize), state)
        @test ℓ.n_gradient_calls == 2
        @test t.stat.n_steps == 1
        @test !hasproperty(state, :κ)
        @test !hasproperty(state, :adaptor)
    end

    @testset "constructors" begin
        rng = Random.default_rng()
        θ_init = randn(rng, 2)
        model = AbstractMCMC.LogDensityModel(CountingLogDensity(0))

        @testset "$T" for T in [Float32, Float64]
            stepsize = PolynomialStepsize(T(0.25), zero(T), one(T))
            for (sampler, metric_type) in (
                (SGLD(stepsize; metric=:unit), UnitEuclideanMetric{T}),
                (SGLD(stepsize; metric=:diagonal), DiagEuclideanMetric{T}),
                (SGLD(stepsize; metric=:dense), DenseEuclideanMetric{T}),
            )
                @test AdvancedHMC.sampler_eltype(sampler) == T
                transition, state = AbstractMCMC.step(
                    rng, model, sampler; n_adapts=0, initial_params=θ_init
                )
                @test eltype(transition.z.θ) == T
                @test eltype(transition.z.r) == T
                @test transition.z.r == zero(transition.z.θ)
                @test transition.stat.n_steps == 1
                @test transition.stat.step_size == stepsize(1)
                @test AdvancedHMC.getmetric(state) isa metric_type
            end
        end
    end

    @testset "weighted estimates" begin
        rng = MersenneTwister(0)
        θ_init = randn(rng, 2)
        model = AdvancedHMC.LogDensityModel(
            LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo)
        )
        sgld_weighted = SGLD(PolynomialStepsize(0.25, 10.0, 0.55); metric=:unit)
        n_sgld_burnin = 20_000
        n_sgld_samples = 60_000
        samples_sgld = AbstractMCMC.sample(
            MersenneTwister(11),
            model,
            sgld_weighted,
            n_sgld_burnin + n_sgld_samples;
            n_adapts=0,
            initial_params=θ_init,
            progress=false,
            verbose=false,
        )
        post_burnin_sgld = samples_sgld[(n_sgld_burnin + 1):end]
        sgld_weight = sum(t.stat.step_size for t in post_burnin_sgld)
        m_est_sgld =
            sum(t.stat.step_size .* invlink_gdemo(t.z.θ) for t in post_burnin_sgld) ./
            sgld_weight

        @test m_est_sgld ≈ [49 / 24, 7 / 6] atol = RNDATOL
    end
end
