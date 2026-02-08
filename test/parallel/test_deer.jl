# Load common test setup (handles both installed package and standalone modes)
include(joinpath(@__DIR__, "common.jl"))

@testset "DEER Algorithm" begin
    @testset "Sequential MCMC Reference" begin
        # Test that sequential_mcmc correctly runs the chain
        D = 3
        T_len = 10

        # Simple linear transition: s_t = 0.9 * s_{t-1} + ω_t
        f(s, ω) = 0.9 .* s .+ ω

        s0 = zeros(D)
        rng = MersenneTwister(42)
        ω = [randn(rng, D) for _ in 1:T_len]

        trajectory = sequential_mcmc(f, s0, T_len, ω)

        @test size(trajectory) == (T_len, D)

        # Verify manually
        s_prev = s0
        for t in 1:T_len
            expected = 0.9 .* s_prev .+ ω[t]
            @test trajectory[t, :] ≈ expected
            s_prev = expected
        end
    end

    @testset "Full DEER - Linear System" begin
        # For a linear system, DEER should converge in 1 iteration
        D = 4
        T_len = 20

        # Linear transition: s_t = A * s_{t-1} + ω_t
        A = 0.5 * I(D) + 0.1 * randn(MersenneTwister(1), D, D)
        f(s, ω) = A * s + ω

        s0 = randn(MersenneTwister(2), D)
        ω = [randn(MersenneTwister(100 + t), D) for t in 1:T_len]

        # Get reference solution
        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Run Full DEER
        result = deer(f, s0, T_len, ω; method=FullDEER(), tol=1e-10, max_iters=10)

        @test result.converged
        @test result.iterations <= 2  # Linear system should converge very fast
        @test result.trajectory ≈ trajectory_ref atol = 1e-8
    end

    @testset "Quasi-DEER - Diagonal Linear System" begin
        # For a diagonal linear system, Quasi-DEER should converge in 1 iteration
        D = 5
        T_len = 30

        # Diagonal transition: s_t = d .* s_{t-1} + ω_t
        d = 0.5 .+ 0.3 .* rand(MersenneTwister(3), D)
        f(s, ω) = d .* s + ω

        s0 = randn(MersenneTwister(4), D)
        ω = [randn(MersenneTwister(200 + t), D) for t in 1:T_len]

        # Get reference solution
        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Run Quasi-DEER
        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=10)

        @test result.converged
        @test result.iterations <= 2  # Diagonal system should converge fast
        @test result.trajectory ≈ trajectory_ref atol = 1e-8
    end

    @testset "Full DEER - Nonlinear System" begin
        # Nonlinear system: s_t = tanh(0.5 * s_{t-1}) + 0.1 * ω_t
        D = 3
        T_len = 15

        f(s, ω) = tanh.(0.5 .* s) .+ 0.1 .* ω

        s0 = randn(MersenneTwister(5), D)
        ω = [randn(MersenneTwister(300 + t), D) for t in 1:T_len]

        # Get reference solution
        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Run Full DEER
        result = deer(f, s0, T_len, ω; method=FullDEER(), tol=1e-10, max_iters=100)

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-8
    end

    @testset "Quasi-DEER - Nonlinear System" begin
        # Quasi-DEER should also converge for contractive nonlinear systems
        D = 4
        T_len = 20

        # Elementwise nonlinear: s_t = 0.5 * s_{t-1}^2 / (1 + s_{t-1}^2) + ω_t
        f(s, ω) = 0.5 .* s .^ 2 ./ (1 .+ s .^ 2) .+ 0.1 .* ω

        s0 = 0.5 .* randn(MersenneTwister(6), D)  # Start small to stay contractive
        ω = [randn(MersenneTwister(400 + t), D) for t in 1:T_len]

        # Get reference solution
        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Run Quasi-DEER (may need more iterations than Full DEER)
        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-8, max_iters=200)

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-6
    end

    @testset "Stochastic Quasi-DEER - Linear System" begin
        # Test stochastic quasi-DEER on a diagonal system
        D = 4
        T_len = 15

        d = 0.6 .+ 0.2 .* rand(MersenneTwister(7), D)
        f(s, ω) = d .* s + ω

        s0 = randn(MersenneTwister(8), D)
        ω = [randn(MersenneTwister(500 + t), D) for t in 1:T_len]

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Stochastic Quasi-DEER with more samples for accuracy
        rng = MersenneTwister(999)
        result = deer(
            f, s0, T_len, ω;
            method=StochasticQuasiDEER(10),  # 10 samples for better estimate
            tol=1e-8,
            max_iters=50,
            rng=rng
        )

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-6
    end

    @testset "DEERResult Fields" begin
        D = 2
        T_len = 5

        f(s, ω) = 0.8 .* s .+ ω
        s0 = zeros(D)
        ω = [randn(MersenneTwister(600 + t), D) for t in 1:T_len]

        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=20)

        @test result isa DEERResult
        @test size(result.trajectory) == (T_len, D)
        @test result.converged isa Bool
        @test result.iterations isa Int
        @test result.iterations > 0
        @test result.max_residual isa Float64
        @test length(result.residual_history) == result.iterations
    end

    @testset "Convergence History" begin
        D = 3
        T_len = 10

        # Use a system that takes more iterations to converge
        # Stronger nonlinearity with weaker contraction
        f(s, ω) = 0.9 .* tanh.(s) .+ 0.1 .* ω
        s0 = 2.0 .* randn(MersenneTwister(9), D)  # Larger initial state
        ω = [randn(MersenneTwister(700 + t), D) for t in 1:T_len]

        result = deer(f, s0, T_len, ω; method=FullDEER(), tol=1e-12, max_iters=100)

        # Check that we have a valid residual history
        @test length(result.residual_history) >= 1
        @test result.max_residual == result.residual_history[end]

        # If we have multiple iterations, residuals should decrease
        if length(result.residual_history) > 1
            @test result.residual_history[end] <= result.residual_history[1]
        end
    end

    @testset "deer_with_settings" begin
        D = 3
        T_len = 10

        f(s, ω) = 0.7 .* s .+ ω
        s0 = zeros(D)
        ω = [randn(MersenneTwister(800 + t), D) for t in 1:T_len]

        settings = ParallelMCMCSettings(method=QuasiDEER(), tol=1e-10, max_iters=50)
        result = deer_with_settings(f, s0, T_len, ω, settings)

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-8
    end

    @testset "Max Iterations Reached" begin
        # Test that max_iters is respected
        D = 2
        T_len = 5

        f(s, ω) = 0.9 .* s .+ ω
        s0 = zeros(D)
        ω = [randn(MersenneTwister(900 + t), D) for t in 1:T_len]

        # Set very tight tolerance that won't be reached in 3 iterations
        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-20, max_iters=3)

        @test result.iterations == 3
        @test length(result.residual_history) == 3
        # May or may not be converged depending on the system
    end

    @testset "Different Methods Agree" begin
        # All methods should converge to the same solution
        D = 3
        T_len = 12

        f(s, ω) = 0.6 .* s .+ 0.1 .* ω
        s0 = randn(MersenneTwister(10), D)
        ω = [randn(MersenneTwister(1000 + t), D) for t in 1:T_len]

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        result_full = deer(f, s0, T_len, ω; method=FullDEER(), tol=1e-10)
        result_quasi = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10)
        result_stoch = deer(
            f, s0, T_len, ω;
            method=StochasticQuasiDEER(20),
            tol=1e-8,
            rng=MersenneTwister(12345)
        )

        @test result_full.trajectory ≈ trajectory_ref atol = 1e-8
        @test result_quasi.trajectory ≈ trajectory_ref atol = 1e-8
        @test result_stoch.trajectory ≈ trajectory_ref atol = 1e-6
    end

    @testset "Longer Chain" begin
        # Test on a longer chain
        D = 2
        T_len = 100

        f(s, ω) = 0.8 .* s .+ 0.2 .* ω
        s0 = zeros(D)
        ω = [randn(MersenneTwister(1100 + t), D) for t in 1:T_len]

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=100)

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-8
    end

    @testset "Identity Transition" begin
        # Edge case: f(s, ω) = s (no dynamics, just initial state propagates)
        D = 3
        T_len = 10

        f(s, ω) = s  # Identity

        s0 = randn(MersenneTwister(11), D)
        ω = [randn(MersenneTwister(1200 + t), D) for t in 1:T_len]  # Unused

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # All rows should equal s0
        for t in 1:T_len
            @test trajectory_ref[t, :] ≈ s0
        end

        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10)
        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-10
    end

    @testset "Pure Noise Transition" begin
        # Edge case: f(s, ω) = ω (state doesn't depend on previous)
        D = 4
        T_len = 8

        f(s, ω) = ω  # Pure noise

        s0 = randn(MersenneTwister(12), D)
        ω = [randn(MersenneTwister(1300 + t), D) for t in 1:T_len]

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        # Each row should equal the corresponding ω
        for t in 1:T_len
            @test trajectory_ref[t, :] ≈ ω[t]
        end

        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10)
        @test result.converged
        @test result.iterations == 1  # Should converge immediately (Jacobian is zero)
        @test result.trajectory ≈ trajectory_ref atol = 1e-10
    end

    @testset "Numerical Stability" begin
        # Test with a system that could have numerical issues
        D = 5
        T_len = 50

        # Nearly contractive system
        f(s, ω) = 0.95 .* s .+ 0.05 .* ω

        s0 = 10.0 .* randn(MersenneTwister(13), D)  # Large initial state
        ω = [randn(MersenneTwister(1400 + t), D) for t in 1:T_len]

        trajectory_ref = sequential_mcmc(f, s0, T_len, ω)

        result = deer(f, s0, T_len, ω; method=QuasiDEER(), tol=1e-10, max_iters=200)

        @test result.converged
        @test result.trajectory ≈ trajectory_ref atol = 1e-7
        @test all(isfinite.(result.trajectory))
    end
end
