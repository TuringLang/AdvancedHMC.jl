using ReTest, Random, AdvancedHMC, AbstractMCMC, ForwardDiff
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD
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

@testset "SGHMC" begin
    @testset "constructor" begin
        @testset "$T" for T in [Float32, Float64]
            sampler = SGHMC(T(0.01), T(0.1), 10)
            @test AdvancedHMC.sampler_eltype(sampler) == T
            @test sampler.n_steps == 10
        end
        @test_throws ArgumentError SGHMC(-0.01, 0.1, 10)
        @test_throws ArgumentError SGHMC(0.01, -0.1, 10)
        @test_throws ArgumentError SGHMC(0.01, 1.1, 10)
        @test_throws ArgumentError SGHMC(0.01, 0.1, 0)
    end

    @testset "getparams and setparams!!" begin
        rng = MersenneTwister(1)
        θ_init = randn(rng, 2)
        new_θ = randn(rng, 2)
        sampler = SGHMC(0.01, 0.1, 100)
        model = AdvancedHMC.LogDensityModel(
            LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo)
        )

        t, state = AbstractMCMC.step(rng, model, sampler; initial_params=θ_init)
        @test AbstractMCMC.getparams(state) == t.z.θ
        new_state = AbstractMCMC.setparams!!(model, state, new_θ)
        @test AbstractMCMC.getparams(new_state) == new_θ
        @test new_state.velocity == state.velocity
        @test new_state.transition.z.r == zero(new_θ)
    end

    @testset "Eq. 15 dynamics" begin
        rng = MersenneTwister(2)
        counted = CountingLogDensity(0)
        counted_model = AbstractMCMC.LogDensityModel(counted)
        counted_sghmc = SGHMC(0.01, 0.1, 4)
        _, counted_state = AbstractMCMC.step(
            rng, counted_model, counted_sghmc; initial_params=zeros(2)
        )
        counted.n_gradient_calls = 0
        counted_transition, _ = AbstractMCMC.step(
            rng, counted_model, counted_sghmc, counted_state
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
        deterministic_state = AdvancedHMC.SGHMCState(
            0, AdvancedHMC.Transition(z, NamedTuple()), v
        )
        deterministic_transition, deterministic_state = AbstractMCMC.step(
            rng, counted_model, deterministic_sghmc, deterministic_state
        )

        @test deterministic_state.velocity == v
        @test deterministic_transition.z.θ == θ .+ deterministic_sghmc.n_steps .* v
        @test deterministic_transition.z.r == zero(θ)
    end

    @testset "AbstractMCMC gdemo sampling" begin
        rng = MersenneTwister(0)
        n_samples = 10_000
        n_adapts = 5_000
        θ_init = randn(rng, 2)
        sampler = SGHMC(0.01, 0.1, 100)
        model = AdvancedHMC.LogDensityModel(
            LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ_gdemo)
        )

        samples = AbstractMCMC.sample(
            rng,
            model,
            sampler,
            n_adapts + n_samples;
            n_adapts=n_adapts,
            initial_params=θ_init,
            progress=false,
            verbose=false,
        )

        for t in samples
            t.z.θ .= invlink_gdemo(t.z.θ)
        end
        m_est = mean(samples) do t
            t.z.θ
        end

        @test m_est ≈ [49 / 24, 7 / 6] atol = RNDATOL
    end
end
