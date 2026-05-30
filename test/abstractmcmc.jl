using ReTest, Random, AdvancedHMC, ForwardDiff, AbstractMCMC
using Statistics: mean

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

@testset "AbstractMCMC w/ gdemo" begin
    rng = MersenneTwister(0)
    n_samples = 10_000
    n_adapts = 5_000
    θ_init = randn(rng, 2)

    nuts = NUTS(0.8)
    hmc = HMC(100; integrator=Leapfrog(0.05))
    hmcda = HMCDA(0.8, 0.1)
    sgld = SGLD(PolynomialStepsize(0.25); metric=:unit)

    integrator = Leapfrog(1e-3)
    κ = AdvancedHMC.make_kernel(nuts, integrator)
    metric = DiagEuclideanMetric(2)
    adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
    custom = HMCSampler(κ, metric, adaptor)

    model = AdvancedHMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo)
    )

    @testset "getparams and setparams!!" begin
        t, s = AbstractMCMC.step(rng, model, nuts;)

        θ = AbstractMCMC.getparams(s)
        @test θ == t.z.θ
        new_state = AbstractMCMC.setparams!!(model, s, θ)
        @test new_state.transition.z.θ == θ
        new_state_logπ = new_state.transition.z.ℓπ
        @test new_state_logπ.value == s.transition.z.ℓπ.value
        @test new_state_logπ.gradient == s.transition.z.ℓπ.gradient
        new_state_logκ = new_state.transition.z.ℓκ
        @test new_state_logκ.value == s.transition.z.ℓκ.value
        @test new_state_logκ.gradient == s.transition.z.ℓκ.gradient
        @test new_state.transition.z.r == s.transition.z.r

        new_θ = randn(rng, 2)
        new_state = AbstractMCMC.setparams!!(model, s, new_θ)
        @test AbstractMCMC.getparams(new_state) == new_θ

        sgld_rng = MersenneTwister(1)
        t, s = AbstractMCMC.step(sgld_rng, model, sgld; initial_params=θ_init)
        @test AbstractMCMC.getparams(s) == t.z.θ
        @test t.z.r == zero(t.z.θ)
        new_state = AbstractMCMC.setparams!!(model, s, new_θ)
        @test AbstractMCMC.getparams(new_state) == new_θ
        @test new_state.transition.z.r == zero(new_θ)
        @test new_state.metric == s.metric
    end

    @testset "SGLD Welling-Teh update" begin
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

    samples_nuts = AbstractMCMC.sample(
        rng,
        model,
        nuts,
        n_adapts + n_samples;
        n_adapts=n_adapts,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )

    # Error if keyword argument `nadapts` is used
    @test_throws ArgumentError AbstractMCMC.sample(
        rng,
        model,
        nuts,
        n_adapts + n_samples;
        nadapts=n_adapts,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )
    @test_throws ArgumentError AbstractMCMC.sample(
        rng,
        model,
        nuts,
        MCMCThreads(),
        n_adapts + n_samples,
        2;
        nadapts=n_adapts,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_nuts
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_nuts = mean(samples_nuts[(n_adapts + 1):end]) do t
        t.z.θ
    end

    @test m_est_nuts ≈ [49 / 24, 7 / 6] atol = RNDATOL

    samples_hmc = AbstractMCMC.sample(
        rng,
        model,
        hmc,
        n_adapts + n_samples;
        n_adapts=n_adapts,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_hmc
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_hmc = mean(samples_hmc) do t
        t.z.θ
    end

    @test m_est_hmc ≈ [49 / 24, 7 / 6] atol = RNDATOL

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

    samples_custom = AbstractMCMC.sample(
        rng,
        model,
        custom,
        n_adapts + n_samples;
        n_adapts=0,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_custom
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_custom = mean(samples_custom[(n_adapts + 1):end]) do t
        t.z.θ
    end

    @test m_est_custom ≈ [49 / 24, 7 / 6] atol = RNDATOL

    # Test that using the same AbstractRNG results in the same chain
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)
    samples1 = AbstractMCMC.sample(
        rng1,
        model,
        custom,
        10;
        n_adapts=0,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )
    samples2 = AbstractMCMC.sample(
        rng2,
        model,
        custom,
        10;
        n_adapts=0,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )
    @test mapreduce(*, samples1, samples2) do s1, s2
        s1.z.θ == s2.z.θ
    end # Equivalent to using all, check that all samples are equal
end
