# Load common test setup (handles both installed package and standalone modes)
include(joinpath(@__DIR__, "common.jl"))

@testset "Jacobian Utilities" begin
    @testset "Finite Difference Jacobian" begin
        # Test with a simple linear function: f(x) = A * x
        D = 4
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)
        J = jacobian_fd(f_linear, x)

        @test size(J) == (D, D)
        @test J ≈ A atol = 1e-5
    end

    @testset "Finite Difference Jacobian - Nonlinear" begin
        # Test with nonlinear function: f(x) = [x[1]^2, x[1]*x[2], sin(x[2])]
        function f_nonlinear(x)
            return [x[1]^2, x[1] * x[2], sin(x[2])]
        end

        x = [2.0, 1.0]
        J = jacobian_fd(f_nonlinear, x)

        # Expected Jacobian:
        # ∂f₁/∂x₁ = 2x₁ = 4,  ∂f₁/∂x₂ = 0
        # ∂f₂/∂x₁ = x₂ = 1,   ∂f₂/∂x₂ = x₁ = 2
        # ∂f₃/∂x₁ = 0,        ∂f₃/∂x₂ = cos(x₂) = cos(1)
        J_expected = [
            4.0 0.0
            1.0 2.0
            0.0 cos(1.0)
        ]

        @test size(J) == (3, 2)
        @test J ≈ J_expected atol = 1e-5
    end

    @testset "Jacobian Diagonal - Linear Function" begin
        # For f(x) = A * x, diag(J) = diag(A)
        D = 5
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)
        d = jacobian_diagonal_full(f_linear, x)

        @test length(d) == D
        @test d ≈ diag(A) atol = 1e-5
    end

    @testset "Jacobian Diagonal - Elementwise Function" begin
        # For f(x) = x.^2, J is diagonal with diag = 2x
        f_square(x) = x .^ 2

        x = rand(5)
        d = jacobian_diagonal_full(f_square, x)

        @test d ≈ 2.0 .* x atol = 1e-5
    end

    @testset "Batch Jacobians" begin
        D = 3
        T_len = 10
        A = rand(D, D)
        f_linear(x) = A * x

        xs = rand(T_len, D)
        Js = batch_jacobians(f_linear, xs)

        @test size(Js) == (T_len, D, D)
        for t in 1:T_len
            @test Js[t, :, :] ≈ A atol = 1e-5
        end
    end

    @testset "Batch Jacobian Diagonals" begin
        D = 4
        T_len = 8

        # f(x) = x.^2, so diag(J) = 2x
        f_square(x) = x .^ 2

        xs = rand(T_len, D)
        diags = batch_jacobian_diagonals(f_square, xs)

        @test size(diags) == (T_len, D)
        for t in 1:T_len
            @test diags[t, :] ≈ 2.0 .* xs[t, :] atol = 1e-5
        end
    end

    @testset "JVP - Finite Difference" begin
        D = 4
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)
        v = rand(D)

        jvp_result = jvp_fd(f_linear, x, v)
        expected = A * v

        @test length(jvp_result) == D
        @test jvp_result ≈ expected atol = 1e-5
    end

    @testset "JVP - Nonlinear Function" begin
        # f(x) = x.^2, J = diag(2x), J*v = 2x .* v
        f_square(x) = x .^ 2

        x = rand(5)
        v = rand(5)

        jvp_result = jvp_fd(f_square, x, v)
        expected = 2.0 .* x .* v

        @test jvp_result ≈ expected atol = 1e-5
    end

    @testset "VJP - Finite Difference" begin
        D = 4
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)
        u = rand(D)

        vjp_result = vjp_fd(f_linear, x, u)
        expected = A' * u

        @test length(vjp_result) == D
        @test vjp_result ≈ expected atol = 1e-5
    end

    @testset "Rademacher Vector" begin
        rng = MersenneTwister(42)
        n = 1000

        z = rademacher_vector(rng, n)

        @test length(z) == n
        @test all(z .== 1.0 .|| z .== -1.0)
        # Check roughly equal split (should be ~500 each)
        @test 400 < sum(z .== 1.0) < 600
    end

    @testset "Hutchinson Diagonal Estimator - Identity" begin
        # For identity function f(x) = x, J = I, diag = ones
        f_identity(x) = x

        D = 10
        x = rand(D)
        rng = MersenneTwister(123)

        # With many samples, should converge to true diagonal
        diag_est = hutchinson_diagonal(f_identity, x, jvp_fd; rng=rng, n_samples=100)

        @test length(diag_est) == D
        @test diag_est ≈ ones(D) atol = 0.2  # Stochastic, so allow some error
    end

    @testset "Hutchinson Diagonal Estimator - Linear" begin
        # For f(x) = A*x where A is diagonal, should recover diag(A)
        D = 8
        diag_A = rand(D)
        A = Diagonal(diag_A)
        f_diag(x) = A * x

        x = rand(D)
        rng = MersenneTwister(456)

        diag_est = hutchinson_diagonal(f_diag, x, jvp_fd; rng=rng, n_samples=200)

        @test diag_est ≈ diag_A atol = 0.15
    end

    @testset "Hutchinson Diagonal Estimator - Elementwise" begin
        # f(x) = x.^2, diag(J) = 2x
        f_square(x) = x .^ 2

        D = 6
        x = rand(D)
        rng = MersenneTwister(789)

        diag_true = 2.0 .* x
        diag_est = hutchinson_diagonal(f_square, x, jvp_fd; rng=rng, n_samples=300)

        @test diag_est ≈ diag_true atol = 0.15
    end

    @testset "Hutchinson Unbiasedness" begin
        # The Hutchinson estimator is unbiased: E[z ⊙ (J*z)] = diag(J)
        # Test that average over many samples approaches true diagonal
        D = 5
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)
        true_diag = diag(A)

        # Run many independent single-sample estimates
        n_trials = 1000
        estimates = zeros(D)
        rng = MersenneTwister(999)

        for _ in 1:n_trials
            est = hutchinson_diagonal(f_linear, x, jvp_fd; rng=rng, n_samples=1)
            estimates .+= est
        end
        estimates ./= n_trials

        # Should be close to true diagonal
        @test estimates ≈ true_diag atol = 0.1
    end

    @testset "Batch Hutchinson Diagonals" begin
        D = 4
        T_len = 5
        f_square(x) = x .^ 2

        xs = rand(T_len, D)
        rng = MersenneTwister(111)

        diags_est = batch_hutchinson_diagonals(f_square, xs, jvp_fd; rng=rng, n_samples=100)

        @test size(diags_est) == (T_len, D)
        for t in 1:T_len
            @test diags_est[t, :] ≈ 2.0 .* xs[t, :] atol = 0.2
        end
    end

    @testset "Hessian Diagonal" begin
        # For log p(x) = -0.5 * x' * x (Gaussian), grad = -x, Hessian = -I
        grad_log_p(x) = -x

        D = 5
        x = rand(D)

        H_diag = hessian_diagonal(grad_log_p, x)

        @test length(H_diag) == D
        @test H_diag ≈ -ones(D) atol = 1e-5
    end

    @testset "Hessian Diagonal - Quadratic" begin
        # log p(x) = -0.5 * x' * Σ⁻¹ * x where Σ⁻¹ is diagonal
        # grad = -Σ⁻¹ * x (for diagonal: -diag(Σ⁻¹) .* x)
        # Hessian = -Σ⁻¹ (diagonal: -diag(Σ⁻¹))
        D = 6
        precision_diag = rand(D) .+ 0.5  # Positive precision

        grad_log_p(x) = -precision_diag .* x

        x = rand(D)
        H_diag = hessian_diagonal(grad_log_p, x)

        @test H_diag ≈ -precision_diag atol = 1e-5
    end

    @testset "Batch Hessian Diagonals" begin
        D = 4
        T_len = 6
        precision_diag = rand(D) .+ 0.5

        grad_log_p(x) = -precision_diag .* x

        xs = rand(T_len, D)
        H_diags = batch_hessian_diagonals(grad_log_p, xs)

        @test size(H_diags) == (T_len, D)
        for t in 1:T_len
            @test H_diags[t, :] ≈ -precision_diag atol = 1e-5
        end
    end

    @testset "Consistency: Full vs Diagonal" begin
        # Verify that jacobian_diagonal_full gives same as diag(jacobian_fd)
        D = 5
        A = rand(D, D)
        f_linear(x) = A * x

        x = rand(D)

        J_full = jacobian_fd(f_linear, x)
        d_from_full = diag(J_full)
        d_direct = jacobian_diagonal_full(f_linear, x)

        @test d_from_full ≈ d_direct atol = 1e-10
    end
end
