using Test
using LinearAlgebra
using Random
using Statistics

# Include the parallel module directly for testing
include(joinpath(@__DIR__, "../../src/parallel/Parallel.jl"))
using .Parallel

@testset "AbstractMCMC Integration" begin
    @testset "SimpleLogDensity" begin
        # Test the SimpleLogDensity wrapper
        D = 3
        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x

        ld = SimpleLogDensity(D, logp, ∇logp)

        @test LogDensityProblems.dimension(ld) == D
        @test LogDensityProblems.capabilities(typeof(ld)) == LogDensityProblems.LogDensityOrder{1}()

        x = randn(D)
        @test LogDensityProblems.logdensity(ld, x) ≈ logp(x)

        lp, grad = LogDensityProblems.logdensity_and_gradient(ld, x)
        @test lp ≈ logp(x)
        @test grad ≈ ∇logp(x)
    end

    @testset "ParallelHMCSampler Construction" begin
        # Test sampler construction
        sampler = ParallelHMCSampler(0.1, 10)

        @test sampler.ε ≈ 0.1
        @test sampler.L == 10
        @test sampler.method isa QuasiDEER
        @test sampler.metric === :diagonal
        @test sampler.tol ≈ 1e-6
        @test sampler.max_iters == 1000

        # Test with custom options
        sampler2 = ParallelHMCSampler(0.05, 20;
            method=FullDEER(),
            metric=:unit,
            tol=1e-8,
            max_iters=500
        )

        @test sampler2.ε ≈ 0.05
        @test sampler2.L == 20
        @test sampler2.method isa FullDEER
        @test sampler2.metric === :unit
        @test sampler2.tol ≈ 1e-8
        @test sampler2.max_iters == 500

        # Test ParallelHMC alias
        sampler3 = ParallelHMC(0.1, 5)
        @test sampler3 isa ParallelHMCSampler
    end

    @testset "ParallelMALASampler Construction" begin
        # Test sampler construction
        sampler = ParallelMALASampler(0.1)

        @test sampler.ε ≈ 0.1
        @test sampler.method isa QuasiDEER
        @test sampler.metric === :diagonal
        @test sampler.tol ≈ 1e-6
        @test sampler.max_iters == 1000

        # Test with custom options
        sampler2 = ParallelMALASampler(0.05;
            method=FullDEER(),
            tol=1e-10
        )

        @test sampler2.ε ≈ 0.05
        @test sampler2.method isa FullDEER
        @test sampler2.tol ≈ 1e-10

        # Test ParallelMALA alias
        sampler3 = ParallelMALA(0.2)
        @test sampler3 isa ParallelMALASampler
    end

    @testset "Parallel Metric Utilities" begin
        D = 5
        T = Float64

        # Test symbol metrics
        M_diag = Parallel.make_parallel_metric(:diagonal, D, T)
        @test M_diag == ones(D)

        M_unit = Parallel.make_parallel_metric(:unit, D, T)
        @test M_unit == ones(D)

        # Test DiagEuclideanMetric
        custom_M = [1.0, 2.0, 0.5, 3.0, 1.5]
        metric = DiagEuclideanMetric{Float64}(custom_M)
        M_from_metric = Parallel.make_parallel_metric(metric, D, T)
        @test M_from_metric == custom_M
    end

    @testset "ParallelSamplerState" begin
        T_len = 10
        D = 3
        trajectory = randn(T_len, D)

        state = ParallelSamplerState(
            trajectory,
            true,   # converged
            5,      # iterations
            1e-8,   # max_residual
            0.85    # acceptance_rate
        )

        @test state.trajectory === trajectory
        @test state.converged == true
        @test state.iterations == 5
        @test state.max_residual ≈ 1e-8
        @test state.acceptance_rate ≈ 0.85
    end

    @testset "get_samples" begin
        T_len = 100
        D = 2
        trajectory = randn(T_len, D)

        state = ParallelSamplerState(
            trajectory, true, 3, 1e-10, 0.9
        )

        # Get all samples
        samples = get_samples(state)
        @test samples === trajectory

        # Get samples after burn-in
        burn_in = 20
        samples_after_burn = get_samples(state, burn_in)
        @test size(samples_after_burn) == (T_len - burn_in, D)
        @test samples_after_burn == trajectory[(burn_in+1):end, :]
    end

    @testset "ParallelSamplerState Iterator" begin
        T_len = 5
        D = 2
        trajectory = [Float64(i+j) for i in 1:T_len, j in 1:D]

        state = ParallelSamplerState(
            trajectory, true, 2, 1e-9, 0.95
        )

        @test length(state) == T_len

        # Test iteration
        samples_collected = []
        for sample in state
            @test sample isa ParallelTransition
            push!(samples_collected, sample.θ)
        end

        @test length(samples_collected) == T_len
        for i in 1:T_len
            @test samples_collected[i] == trajectory[i, :]
        end
    end

    @testset "Parallel HMC Sampling - Gaussian" begin
        rng = MersenneTwister(42)
        D = 2
        N = 50

        # Standard Gaussian
        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelHMCSampler(0.1, 10; tol=1e-6, max_iters=100)

        state = parallel_sample(rng, ld, sampler, N; initial_params=zeros(D))

        @test state isa ParallelSamplerState
        @test size(state.trajectory) == (N, D)
        @test state.converged
        @test all(isfinite.(state.trajectory))
    end

    @testset "Parallel MALA Sampling - Gaussian" begin
        rng = MersenneTwister(123)
        D = 2
        N = 50

        # Standard Gaussian
        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelMALASampler(0.1; tol=1e-6, max_iters=100)

        state = parallel_sample(rng, ld, sampler, N; initial_params=zeros(D))

        @test state isa ParallelSamplerState
        @test size(state.trajectory) == (N, D)
        @test state.converged
        @test all(isfinite.(state.trajectory))
    end

    @testset "Parallel HMC Sample Statistics" begin
        rng = MersenneTwister(456)
        D = 2
        N = 200

        # Standard Gaussian - samples should have mean ~0, var ~1
        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelHMCSampler(0.2, 15; tol=1e-6)

        state = parallel_sample(rng, ld, sampler, N; initial_params=zeros(D))

        # Discard burn-in
        samples = get_samples(state, 50)

        # Check sample statistics
        sample_mean = vec(mean(samples, dims=1))
        sample_var = vec(var(samples, dims=1))

        # Should be approximately 0 mean, 1 variance (loose bounds for finite samples)
        @test all(abs.(sample_mean) .< 1.0)
        @test all(0.1 .< sample_var .< 5.0)
    end

    @testset "Parallel MALA Sample Statistics" begin
        rng = MersenneTwister(789)
        D = 2
        N = 200

        # Standard Gaussian
        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelMALASampler(0.2; tol=1e-6)

        state = parallel_sample(rng, ld, sampler, N; initial_params=zeros(D))

        # Discard burn-in
        samples = get_samples(state, 50)

        # Check sample statistics
        sample_mean = vec(mean(samples, dims=1))
        sample_var = vec(var(samples, dims=1))

        # Should be approximately 0 mean, 1 variance (loose bounds for finite samples)
        @test all(abs.(sample_mean) .< 1.0)
        @test all(0.1 .< sample_var .< 5.0)
    end

    @testset "Parallel HMC with Different Methods" begin
        rng = MersenneTwister(101112)
        D = 2
        N = 30

        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        # Test with QuasiDEER
        sampler_quasi = ParallelHMCSampler(0.1, 5; method=QuasiDEER(), tol=1e-6)
        state_quasi = parallel_sample(rng, ld, sampler_quasi, N; initial_params=zeros(D))
        @test state_quasi.converged

        # Test with FullDEER
        sampler_full = ParallelHMCSampler(0.1, 5; method=FullDEER(), tol=1e-6)
        state_full = parallel_sample(MersenneTwister(101112), ld, sampler_full, N; initial_params=zeros(D))
        @test state_full.converged
    end

    @testset "Parallel Sampling without explicit RNG" begin
        D = 2
        N = 20

        logp = x -> -0.5 * sum(x .^ 2)
        ∇logp = x -> -x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelHMCSampler(0.1, 5; tol=1e-6)

        # Call without RNG
        state = parallel_sample(ld, sampler, N; initial_params=zeros(D))

        @test state isa ParallelSamplerState
        @test size(state.trajectory) == (N, D)
    end

    @testset "Non-isotropic Gaussian" begin
        rng = MersenneTwister(131415)
        D = 3
        N = 100

        # Non-isotropic Gaussian with precision [1, 2, 0.5]
        precision = [1.0, 2.0, 0.5]
        logp = x -> -0.5 * sum(precision .* x .^ 2)
        ∇logp = x -> -precision .* x
        ld = SimpleLogDensity(D, logp, ∇logp)

        sampler = ParallelHMCSampler(0.15, 10; tol=1e-6)
        state = parallel_sample(rng, ld, sampler, N; initial_params=zeros(D))

        @test state.converged

        # Samples should have variance 1/precision
        samples = get_samples(state, 30)
        sample_var = vec(var(samples, dims=1))
        expected_var = 1.0 ./ precision

        # Loose check due to finite samples
        for i in 1:D
            @test 0.2 * expected_var[i] < sample_var[i] < 5.0 * expected_var[i]
        end
    end

    @testset "AbstractParallelSampler Type" begin
        sampler_hmc = ParallelHMCSampler(0.1, 10)
        sampler_mala = ParallelMALASampler(0.1)

        @test sampler_hmc isa AbstractParallelSampler
        @test sampler_mala isa AbstractParallelSampler
    end

    @testset "ParallelTransition" begin
        θ = [1.0, 2.0, 3.0]
        stat = (iteration=5, converged=true)

        trans = ParallelTransition(θ, stat)

        @test trans.θ == θ
        @test trans.stat.iteration == 5
        @test trans.stat.converged == true
    end
end
