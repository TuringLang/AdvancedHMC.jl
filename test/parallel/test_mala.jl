# Load common test setup (handles both installed package and standalone modes)
include(joinpath(@__DIR__, "common.jl"))

@testset "Parallel MALA" begin
    @testset "MALARandomInputs" begin
        rng = MersenneTwister(42)
        D = 5
        T_len = 10

        ω = sample_mala_inputs(rng, D, T_len)

        @test length(ω) == T_len
        @test all(length(ω[t].ξ) == D for t in 1:T_len)
        @test all(0 < ω[t].u < 1 for t in 1:T_len)
    end

    @testset "MALA Proposal" begin
        D = 3
        x = randn(MersenneTwister(1), D)
        ∇logp_x = -x  # Gradient for Gaussian
        ε = 0.1
        ξ = randn(MersenneTwister(2), D)

        x̃ = mala_proposal(x, ∇logp_x, ε, ξ)

        expected = x .+ ε .* ∇logp_x .+ sqrt(2 * ε) .* ξ
        @test x̃ ≈ expected
    end

    @testset "Sequential MALA - Gaussian" begin
        # Test MALA on a standard Gaussian target
        D = 2
        T_len = 100

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.5
        s0 = zeros(D)

        rng = MersenneTwister(123)
        trajectory, acceptance_rate = sequential_mala(logp, ∇logp, ε, s0, T_len; rng=rng)

        @test size(trajectory) == (T_len, D)
        @test 0.0 <= acceptance_rate <= 1.0

        # Acceptance rate should be reasonable for well-tuned ε
        @test acceptance_rate > 0.3
    end

    @testset "Parallel MALA Matches Sequential - Gaussian" begin
        D = 3
        T_len = 50

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.3
        s0 = randn(MersenneTwister(10), D)

        # Sample random inputs
        rng = MersenneTwister(456)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        # Run sequential
        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        # Run parallel MALA with DEER
        result = parallel_mala(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=200)

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-6
    end

    @testset "Parallel MALA - Full DEER" begin
        D = 2
        T_len = 30

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.2
        s0 = randn(MersenneTwister(20), D)

        rng = MersenneTwister(789)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        result = parallel_mala(config, s0, T_len, ω; method=FullDEER(), tol=1e-10, max_iters=100)

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-6
    end

    @testset "Parallel MALA - Stochastic Quasi-DEER" begin
        D = 2
        T_len = 20

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.2
        s0 = randn(MersenneTwister(30), D)

        rng = MersenneTwister(101112)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        # Stochastic method needs more samples for accuracy
        result = parallel_mala(
            config, s0, T_len, ω;
            method=StochasticQuasiDEER(20),
            tol=1e-8,
            max_iters=200,
            rng=MersenneTwister(999)
        )

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-5
    end

    @testset "Parallel MALA - Diagonal Gaussian" begin
        # Test with non-isotropic Gaussian
        D = 4
        T_len = 40

        # Diagonal precision matrix
        precision = [1.0, 2.0, 0.5, 1.5]

        logp(x) = -0.5 * sum(precision .* x .^ 2)
        ∇logp(x) = -precision .* x

        ε = 0.15
        s0 = randn(MersenneTwister(40), D)

        rng = MersenneTwister(131415)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        result = parallel_mala(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-10)

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-6
    end

    @testset "Parallel MALA - Correlated Gaussian" begin
        # Test with correlated Gaussian (full covariance)
        D = 3
        T_len = 30

        # Create a positive definite precision matrix
        A = randn(MersenneTwister(50), D, D)
        Σ_inv = A' * A + I(D)  # Ensure positive definite

        logp(x) = -0.5 * dot(x, Σ_inv * x)
        ∇logp(x) = -Σ_inv * x

        ε = 0.1
        s0 = randn(MersenneTwister(51), D)

        rng = MersenneTwister(161718)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        # Full DEER should work well for correlated targets
        result = parallel_mala(config, s0, T_len, ω; method=FullDEER(), tol=1e-10)

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-6
    end

    @testset "Parallel MALA Convenience API" begin
        D = 2
        T_len = 25

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.3
        s0 = zeros(D)

        # Use convenience API that samples inputs automatically
        result = parallel_mala(
            logp, ∇logp, ε, s0, T_len;
            rng=MersenneTwister(192021),
            method=QuasiDEER(),
            tol=1e-10
        )

        @test result isa DEERResult
        @test size(result.trajectory) == (T_len, D)
        @test result.converged
    end

    @testset "MALAConfig" begin
        logp(x) = -sum(x .^ 2)
        ∇logp(x) = -2 .* x
        ε = 0.1

        config = MALAConfig(ε, logp, ∇logp)

        @test config.ε == 0.1
        @test config.logp([1.0, 2.0]) == logp([1.0, 2.0])
        @test config.∇logp([1.0, 2.0]) == ∇logp([1.0, 2.0])
    end

    @testset "Acceptance Behavior" begin
        # Test that acceptance/rejection works correctly
        # MH accepts when u < α, i.e., log(u) < log(α)
        # - Small u (near 0) → log(u) very negative → easier to accept
        # - Large u (near 1) → log(u) near 0 → harder to accept
        D = 2

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        x = [0.0, 0.0]

        # Test acceptance: proposal from mode with small u (easy to accept)
        ξ_small = zeros(D)  # Proposal stays near mode
        ω_accept = MALARandomInputs(ξ_small, 0.001)  # Very small u → accept

        x_new = mala_transition(x, ω_accept, logp, ∇logp, ε)
        # Should accept (move slightly due to gradient term which is 0 at origin)
        @test x_new ≈ x .+ ε .* ∇logp(x) .+ sqrt(2 * ε) .* ξ_small atol = 1e-10

        # Test rejection: proposal far from mode with large u (hard to accept)
        ξ_large = [10.0, 10.0]  # Proposal goes far from mode
        ω_reject = MALARandomInputs(ξ_large, 0.9999)  # Very large u → reject

        x_new = mala_transition(x, ω_reject, logp, ∇logp, ε)
        # Should reject (stay at x) because α will be very small for bad proposal
        @test x_new ≈ x atol = 1e-10
    end

    @testset "Longer Chain" begin
        D = 2
        T_len = 200

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.5
        s0 = [3.0, -2.0]  # Start away from mode

        rng = MersenneTwister(222324)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        traj_seq, _ = sequential_mala(config, s0, T_len, ω)

        result = parallel_mala(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=500)

        @test result.converged
        @test result.trajectory ≈ traj_seq atol = 1e-6
    end

    @testset "Sample Statistics" begin
        # Run longer chain and check sample statistics approximate target
        D = 2
        T_len = 500

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.8  # Larger step for faster mixing
        s0 = zeros(D)

        rng = MersenneTwister(252627)

        # Use sequential for this test (faster, and we just want to test MALA correctness)
        trajectory, acceptance_rate = sequential_mala(logp, ∇logp, ε, s0, T_len; rng=rng)

        # Discard burn-in
        burn_in = 100
        samples = trajectory[(burn_in + 1):end, :]

        # Sample mean should be close to 0
        sample_mean = vec(mean(samples, dims=1))
        @test all(abs.(sample_mean) .< 0.3)  # Loose bound due to finite samples

        # Sample variance should be close to 1
        sample_var = vec(var(samples, dims=1))
        @test all(0.5 .< sample_var .< 2.0)  # Loose bounds
    end

    @testset "Numerical Stability" begin
        D = 3
        T_len = 50

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.2
        s0 = 5.0 .* randn(MersenneTwister(60), D)  # Start far from mode

        rng = MersenneTwister(282930)
        ω = sample_mala_inputs(rng, D, T_len)

        config = MALAConfig(ε, logp, ∇logp)

        result = parallel_mala(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-10)

        @test result.converged
        @test all(isfinite.(result.trajectory))
    end
end
