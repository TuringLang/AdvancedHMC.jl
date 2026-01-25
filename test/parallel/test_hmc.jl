using Test
using LinearAlgebra
using Random
using Statistics

# Include the parallel module directly for testing
include(joinpath(@__DIR__, "../../src/parallel/Parallel.jl"))
using .Parallel

@testset "Parallel HMC" begin
    @testset "HMCRandomInputs" begin
        rng = MersenneTwister(42)
        D = 5
        T_len = 10

        ω = sample_hmc_inputs(rng, D, T_len)

        @test length(ω) == T_len
        @test all(length(ω[t].r) == D for t in 1:T_len)
        @test all(0 < ω[t].u < 1 for t in 1:T_len)
    end

    @testset "HMCRandomInputs with Mass Matrix" begin
        rng = MersenneTwister(42)
        D = 3
        T_len = 100
        M⁻¹ = [1.0, 0.5, 2.0]  # Inverse mass matrix

        ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)

        # Check that momentum has correct variance
        # r ~ N(0, M) so Var(r_i) = M_i = 1/M⁻¹_i
        r_samples = hcat([ω[t].r for t in 1:T_len]...)'
        r_vars = vec(var(r_samples, dims=1))

        # Loose check (finite samples)
        expected_vars = 1 ./ M⁻¹
        @test all(0.3 .* expected_vars .< r_vars .< 3.0 .* expected_vars)
    end

    @testset "Leapfrog Step" begin
        D = 3
        θ = [1.0, 2.0, 3.0]
        r = [0.1, -0.2, 0.3]
        ∇logp = -θ  # Gradient for Gaussian
        ε = 0.1
        M⁻¹ = ones(D)

        θ_new, r_half = leapfrog_step(θ, r, ∇logp, ε; M⁻¹=M⁻¹)

        # Check half-step for momentum: r_half = r + ε/2 * ∇logp
        expected_r_half = r .+ (ε / 2) .* ∇logp
        @test r_half ≈ expected_r_half

        # Check full step for position: θ_new = θ + ε * M⁻¹ * r_half
        expected_θ = θ .+ ε .* (M⁻¹ .* expected_r_half)
        @test θ_new ≈ expected_θ
    end

    @testset "Leapfrog Full Integration" begin
        D = 2
        θ = [2.0, -1.0]
        r = [0.5, 0.3]
        ε = 0.1
        L = 10
        M⁻¹ = ones(D)

        # Gaussian target
        ∇logp(x) = -x

        θ_final, r_final = leapfrog_full(θ, r, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Should be finite
        @test all(isfinite.(θ_final))
        @test all(isfinite.(r_final))

        # Energy should be approximately conserved (for small ε)
        H_init = 0.5 * sum(θ .^ 2) + 0.5 * sum(r .^ 2)
        H_final = 0.5 * sum(θ_final .^ 2) + 0.5 * sum(r_final .^ 2)
        @test abs(H_final - H_init) < 0.5  # Loose bound
    end

    @testset "HMC Proposal" begin
        D = 2
        θ = [1.0, -0.5]
        r = [0.3, 0.2]
        ε = 0.1
        L = 5
        M⁻¹ = ones(D)

        ∇logp(x) = -x

        θ_prop, r_prop = hmc_proposal(θ, r, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Momentum should be negated for reversibility
        θ_check, r_check = leapfrog_full(θ, r, ∇logp, ε, L; M⁻¹=M⁻¹)
        @test θ_prop ≈ θ_check
        @test r_prop ≈ -r_check
    end

    @testset "HMC Transition - Accept" begin
        D = 2
        rng = MersenneTwister(123)

        # Gaussian target
        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        θ = zeros(D)  # At mode
        ε = 0.1
        L = 10
        M⁻¹ = ones(D)

        # Small momentum + small u should accept
        ω = HMCRandomInputs(0.1 .* randn(rng, D), 0.001)

        θ_new = hmc_transition(θ, ω, logp, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Should have moved (accepted)
        @test !all(θ_new .≈ θ)
    end

    @testset "Sequential HMC - Gaussian" begin
        D = 2
        T_len = 100

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        L = 10
        s0 = zeros(D)

        rng = MersenneTwister(456)
        trajectory, acceptance_rate = sequential_hmc(logp, ∇logp, ε, L, s0, T_len; rng=rng)

        @test size(trajectory) == (T_len, D)
        @test 0.0 <= acceptance_rate <= 1.0

        # HMC should have good acceptance for well-tuned parameters
        @test acceptance_rate > 0.5
    end

    @testset "Parallel HMC Matches Sequential - Gaussian" begin
        D = 2
        T_len = 30

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        L = 5
        s0 = randn(MersenneTwister(10), D)
        M⁻¹ = ones(D)

        # Sample random inputs
        rng = MersenneTwister(789)
        ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)

        config = HMCConfig(ε, L, logp, ∇logp, M⁻¹)

        # Run sequential
        traj_seq, _ = sequential_hmc(config, s0, T_len, ω)

        # Run parallel HMC with DEER
        # Note: soft gating means slight differences are expected
        result = parallel_hmc(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-8, max_iters=200)

        @test result.converged
        # Allow larger tolerance due to soft gating
        @test maximum(abs.(result.trajectory .- traj_seq)) < 0.5
    end

    @testset "Parallel HMC - Full DEER" begin
        D = 2
        T_len = 20

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        L = 3
        s0 = randn(MersenneTwister(20), D)
        M⁻¹ = ones(D)

        rng = MersenneTwister(101112)
        ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)

        config = HMCConfig(ε, L, logp, ∇logp, M⁻¹)

        traj_seq, _ = sequential_hmc(config, s0, T_len, ω)

        result = parallel_hmc(config, s0, T_len, ω; method=FullDEER(), tol=1e-8, max_iters=100)

        @test result.converged
        @test maximum(abs.(result.trajectory .- traj_seq)) < 0.5
    end

    @testset "Parallel HMC Convenience API" begin
        D = 2
        T_len = 25

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        L = 5
        s0 = zeros(D)

        result = parallel_hmc(
            logp, ∇logp, ε, L, s0, T_len;
            rng=MersenneTwister(131415),
            method=QuasiDEER(),
            tol=1e-8
        )

        @test result isa DEERResult
        @test size(result.trajectory) == (T_len, D)
        @test result.converged
    end

    @testset "HMCConfig" begin
        logp(x) = -sum(x .^ 2)
        ∇logp(x) = -2 .* x
        ε = 0.1
        L = 10

        config = HMCConfig(ε, L, logp, ∇logp, 3)

        @test config.ε == 0.1
        @test config.L == 10
        @test config.logp([1.0, 2.0, 3.0]) == logp([1.0, 2.0, 3.0])
        @test length(config.M⁻¹) == 3
    end

    @testset "Leapfrog Transition for DEER" begin
        D = 2
        ε = 0.1
        M⁻¹ = ones(D)

        ∇logp(x) = -x

        # Initial state [θ; r]
        θ0 = [1.0, -0.5]
        r0 = [0.2, 0.3]
        s0 = vcat(θ0, r0)

        # One leapfrog step
        s_new = leapfrog_transition(s0, nothing, ∇logp, ε, M⁻¹)

        @test length(s_new) == 2 * D
        @test all(isfinite.(s_new))

        # Extract θ and r
        θ_new = s_new[1:D]
        r_new = s_new[(D+1):end]

        # Verify manually
        grad0 = ∇logp(θ0)
        r_half = r0 .+ (ε / 2) .* grad0
        θ_exp = θ0 .+ ε .* (M⁻¹ .* r_half)
        grad_new = ∇logp(θ_exp)
        r_exp = r_half .+ (ε / 2) .* grad_new

        @test θ_new ≈ θ_exp
        @test r_new ≈ r_exp
    end

    @testset "Parallel Leapfrog" begin
        D = 2
        L = 10
        ε = 0.1
        M⁻¹ = ones(D)

        ∇logp(x) = -x

        θ0 = [1.0, -0.5]
        r0 = [0.2, 0.3]

        # Sequential leapfrog
        θ_seq, r_seq = leapfrog_full(θ0, r0, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Parallel leapfrog
        result = parallel_leapfrog(θ0, r0, ∇logp, ε, L, M⁻¹; method=QuasiDEER(), tol=1e-10)

        @test result.converged

        # Extract final state
        s_final = result.trajectory[end, :]
        θ_par = s_final[1:D]
        r_par = s_final[(D+1):end]

        @test θ_par ≈ θ_seq atol = 1e-6
        @test r_par ≈ r_seq atol = 1e-6
    end

    @testset "HMC Energy Conservation" begin
        # Test that leapfrog approximately conserves energy
        D = 3
        L = 50
        ε = 0.05
        M⁻¹ = ones(D)

        rng = MersenneTwister(999)
        θ0 = randn(rng, D)
        r0 = randn(rng, D)

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        H_init = -logp(θ0) + 0.5 * sum(M⁻¹ .* r0 .^ 2)

        θ_final, r_final = leapfrog_full(θ0, r0, ∇logp, ε, L; M⁻¹=M⁻¹)

        H_final = -logp(θ_final) + 0.5 * sum(M⁻¹ .* r_final .^ 2)

        # Energy should be well conserved for small step size
        @test abs(H_final - H_init) / abs(H_init) < 0.1
    end

    @testset "Diagonal Mass Matrix" begin
        D = 3
        T_len = 50

        # Non-isotropic Gaussian
        precision = [1.0, 2.0, 0.5]

        logp(x) = -0.5 * sum(precision .* x .^ 2)
        ∇logp(x) = -precision .* x

        # Use optimal mass matrix
        M⁻¹ = precision  # M⁻¹ = Σ⁻¹ for target with precision Σ⁻¹

        ε = 0.2
        L = 10
        s0 = zeros(D)

        rng = MersenneTwister(161718)
        trajectory, acceptance_rate = sequential_hmc(logp, ∇logp, ε, L, s0, T_len; rng=rng, M⁻¹=M⁻¹)

        @test size(trajectory) == (T_len, D)
        @test acceptance_rate > 0.5
    end

    @testset "Hessian Diagonal FD" begin
        # Test Hessian diagonal computation
        D = 3

        # Quadratic function: logp(x) = -0.5 * x' * A * x
        # Hessian of logp = -A
        A = [2.0, 1.0, 3.0]  # Diagonal

        logp(x) = -0.5 * sum(A .* x .^ 2)
        ∇logp(x) = -A .* x

        θ = [1.0, -0.5, 2.0]
        H_diag = hessian_diagonal_fd(∇logp, θ)

        # Should be -A (Hessian of logp)
        @test H_diag ≈ -A atol = 1e-4
    end

    @testset "Sample Statistics" begin
        # Run HMC and check sample statistics
        D = 2
        T_len = 500

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.3
        L = 20
        s0 = zeros(D)

        rng = MersenneTwister(192021)
        trajectory, acceptance_rate = sequential_hmc(logp, ∇logp, ε, L, s0, T_len; rng=rng)

        # Discard burn-in
        burn_in = 100
        samples = trajectory[(burn_in+1):end, :]

        # Sample mean should be close to 0
        sample_mean = vec(mean(samples, dims=1))
        @test all(abs.(sample_mean) .< 0.3)

        # Sample variance should be close to 1
        sample_var = vec(var(samples, dims=1))
        @test all(0.5 .< sample_var .< 2.0)
    end

    @testset "Numerical Stability" begin
        D = 3
        T_len = 30

        logp(x) = -0.5 * sum(x .^ 2)
        ∇logp(x) = -x

        ε = 0.1
        L = 5
        s0 = 5.0 .* randn(MersenneTwister(60), D)  # Start far from mode
        M⁻¹ = ones(D)

        rng = MersenneTwister(222324)
        ω = sample_hmc_inputs(rng, D, T_len; M⁻¹=M⁻¹)

        config = HMCConfig(ε, L, logp, ∇logp, M⁻¹)

        result = parallel_hmc(config, s0, T_len, ω; method=QuasiDEER(), tol=1e-8)

        @test result.converged
        @test all(isfinite.(result.trajectory))
    end

    @testset "Block Quasi-DEER for Leapfrog" begin
        D = 2
        L = 10
        ε = 0.1
        M⁻¹ = ones(D)

        # Gaussian target
        ∇logp(x) = -x
        hessian_diag(x) = -ones(D)  # Hessian of -logp = -I

        θ0 = [1.0, -0.5]
        r0 = [0.2, 0.3]
        s0 = vcat(θ0, r0)

        # Sequential leapfrog
        θ_seq, r_seq = leapfrog_full(θ0, r0, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Parallel leapfrog with Block Quasi-DEER
        method = BlockQuasiDEER(hessian_diag, ε, M⁻¹)

        # Create dummy transition function
        f(s, ω_t) = leapfrog_transition(s, ω_t, ∇logp, ε, M⁻¹)
        ω = [nothing for _ in 1:L]

        result = deer(f, s0, L, ω; method=method, tol=1e-10, max_iters=100)

        @test result.converged

        # Extract final state
        s_final = result.trajectory[end, :]
        θ_par = s_final[1:D]
        r_par = s_final[(D+1):end]

        @test θ_par ≈ θ_seq atol = 1e-6
        @test r_par ≈ r_seq atol = 1e-6
    end

    @testset "Block Quasi-DEER - Non-isotropic" begin
        D = 3
        L = 15
        ε = 0.08
        M⁻¹ = [1.0, 0.5, 2.0]

        # Non-isotropic Gaussian
        precision = [1.0, 2.0, 0.5]
        ∇logp(x) = -precision .* x
        hessian_diag(x) = -precision  # Constant Hessian for quadratic

        rng = MersenneTwister(252627)
        θ0 = randn(rng, D)
        r0 = randn(rng, D)
        s0 = vcat(θ0, r0)

        # Sequential
        θ_seq, r_seq = leapfrog_full(θ0, r0, ∇logp, ε, L; M⁻¹=M⁻¹)

        # Block Quasi-DEER
        method = BlockQuasiDEER(hessian_diag, ε, M⁻¹)
        f(s, ω_t) = leapfrog_transition(s, ω_t, ∇logp, ε, M⁻¹)
        ω = [nothing for _ in 1:L]

        result = deer(f, s0, L, ω; method=method, tol=1e-10, max_iters=100)

        @test result.converged

        s_final = result.trajectory[end, :]
        @test s_final[1:D] ≈ θ_seq atol = 1e-5
        @test s_final[(D+1):end] ≈ r_seq atol = 1e-5
    end

    @testset "Block Quasi-DEER Convergence" begin
        D = 2
        L = 20
        ε = 0.1
        M⁻¹ = ones(D)

        ∇logp(x) = -x
        hessian_diag(x) = -ones(D)

        θ0 = [2.0, -1.0]
        r0 = [0.5, 0.3]
        s0 = vcat(θ0, r0)

        method = BlockQuasiDEER(hessian_diag, ε, M⁻¹)
        f(s, ω_t) = leapfrog_transition(s, ω_t, ∇logp, ε, M⁻¹)
        ω = [nothing for _ in 1:L]

        result = deer(f, s0, L, ω; method=method, tol=1e-12, max_iters=50)

        @test result.converged
        # Should converge in few iterations for Gaussian
        @test result.iterations <= 5
    end
end
