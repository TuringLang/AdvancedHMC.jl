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
    sghmc = SGHMC(0.01, 0.1, 100)

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

        sghmc_rng = MersenneTwister(1)
        t, s = AbstractMCMC.step(sghmc_rng, model, sghmc; initial_params=θ_init)
        @test AbstractMCMC.getparams(s) == t.z.θ
        new_state = AbstractMCMC.setparams!!(model, s, new_θ)
        @test AbstractMCMC.getparams(new_state) == new_θ
        @test new_state.velocity == s.velocity
        @test new_state.transition.z.r == zero(new_θ)
    end

    @testset "SGHMC Eq. 15 dynamics" begin
        sghmc_rng = MersenneTwister(2)
        counted = CountingLogDensity(0)
        counted_model = AbstractMCMC.LogDensityModel(counted)
        counted_sghmc = SGHMC(0.01, 0.1, 4)
        _, counted_state = AbstractMCMC.step(
            sghmc_rng, counted_model, counted_sghmc; initial_params=zeros(2)
        )
        counted.n_gradient_calls = 0
        counted_transition, _ = AbstractMCMC.step(
            sghmc_rng, counted_model, counted_sghmc, counted_state
        )

        @test counted.n_gradient_calls == counted_sghmc.n_steps
        @test counted_transition.stat.is_accept == true
        @test counted_transition.stat.acceptance_rate == 1
        @test counted_transition.stat.is_adapt == false
        @test counted_transition.stat.numerical_error == false

        deterministic_sghmc = SGHMC(0.0, 0.0, 3)
        θ = [1.0, 2.0]
        v = [0.25, -0.5]
        h = AdvancedHMC.Hamiltonian(UnitEuclideanMetric(2), counted_model)
        z = AdvancedHMC.phasepoint(h, θ, zero(θ))
        deterministic_state = AdvancedHMC.SGHMCState(0, AdvancedHMC.Transition(z, NamedTuple()), v)
        deterministic_transition, deterministic_state = AbstractMCMC.step(
            sghmc_rng, counted_model, deterministic_sghmc, deterministic_state
        )

        @test deterministic_state.velocity == v
        @test deterministic_transition.z.θ == θ .+ deterministic_sghmc.n_steps .* v
        @test deterministic_transition.z.r == zero(θ)
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

    samples_sghmc = AbstractMCMC.sample(
        rng,
        model,
        sghmc,
        n_adapts + n_samples;
        n_adapts=n_adapts,
        initial_params=θ_init,
        progress=false,
        verbose=false,
    )

    # Transform back to original space.
    # NOTE: We're not correcting for the `logabsdetjac` here since, but
    # we're only interested in the mean it doesn't matter.
    for t in samples_sghmc
        t.z.θ .= invlink_gdemo(t.z.θ)
    end
    m_est_sghmc = mean(samples_sghmc) do t
        t.z.θ
    end

    @test m_est_sghmc ≈ [49 / 24, 7 / 6] atol = RNDATOL

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
