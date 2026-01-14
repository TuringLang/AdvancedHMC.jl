using ReTest, Random
using AdvancedHMC, ForwardDiff, AbstractMCMC
using LinearAlgebra
using Distributions: MvNormal, logpdf
using MCMCLogDensityProblems
using FiniteDiff:
    finite_difference_gradient, finite_difference_hessian, finite_difference_jacobian
using AdvancedHMC:
    neg_energy,
    energy,
    ∂H∂θ,
    ∂H∂r,
    metric_eval,
    metric_sensitivity,
    logdet_grad_matrix,
    kinetic_grad_matrix,
    SoftAbsEval,
    RiemannianMetric,
    SoftAbsRiemannianMetric
using Statistics

####
#### Test utilities
####

function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    _Hfunc = gen_hess_fwd(Vfunc, x)
    Hfunc = x -> _Hfunc(x)[3]
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[((i - 1) * d + 1):(i * d), :] for i in 1:d)...; dims=3)
end

function prepare_sample(ℓπ, initial_θ, λ)
    Vfunc = x -> -ℓπ(x)
    _Hfunc = MCMCLogDensityProblems.gen_hess(Vfunc, initial_θ)
    Hfunc = x -> copy.(_Hfunc(x))

    fstabilize = H -> H + λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(x -> -ℓπ(x), initial_θ; f=fstabilize)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x))

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

δ(a, b) = maximum(abs.(a - b))

####
#### Tests for unified API (RiemannianMetric, SoftAbsRiemannianMetric)
####

@testset "New Riemannian API" begin
    @testset "$(nameof(typeof(target)))" for target in [HighDimGaussian(2), Funnel()]
        rng = MersenneTwister(1110)
        λ = 1e-2

        θ₀ = rand(rng, dim(target))

        ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
        ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, θ₀)

        _, _, Gfunc, ∂G∂θfunc = prepare_sample(ℓπ, θ₀, λ)

        D = dim(target)
        x = zeros(D)
        r = randn(rng, D)

        @testset "RiemannianMetric (PDMat-style)" begin
            metric = RiemannianMetric((D,), Gfunc, ∂G∂θfunc)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            # Test metric_eval returns a matrix
            G_eval = metric_eval(metric, x)
            @test G_eval isa AbstractMatrix
            @test size(G_eval) == (D, D)

            # Test metric_sensitivity
            ∂G = metric_sensitivity(metric, x)
            @test size(∂G) == (D, D, D)

            # Test gradient matrices
            M_logdet = logdet_grad_matrix(G_eval)
            @test size(M_logdet) == (D, D)

            M_kinetic = kinetic_grad_matrix(G_eval, r)
            @test size(M_kinetic) == (D, D)

            # Test ∂H∂θ against finite differences
            Hamifunc = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
            Hamifuncx = x -> Hamifunc(x, r)
            @test δ(
                finite_difference_gradient(Hamifuncx, x), ∂H∂θ(hamiltonian, x, r).gradient
            ) < 1e-4

            # Test ∂H∂r against finite differences
            Hamifuncr = r -> Hamifunc(x, r)
            @test δ(finite_difference_gradient(Hamifuncr, r), ∂H∂r(hamiltonian, x, r)) <
                1e-4
        end

        @testset "SoftAbsRiemannianMetric" begin
            α = 20.0
            metric = SoftAbsRiemannianMetric((D,), Gfunc, ∂G∂θfunc, α)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            # Test metric_eval returns SoftAbsEval
            G_eval = metric_eval(metric, x)
            @test G_eval isa SoftAbsEval
            @test size(G_eval.Q) == (D, D)
            @test length(G_eval.softabsλ) == D
            @test size(G_eval.J) == (D, D)
            @test size(G_eval.M_logdet) == (D, D)

            # Test standard operations on SoftAbsEval
            v = randn(rng, D)
            @test length(G_eval \ v) == D
            @test logdet(G_eval) isa Real

            # Test gradient matrices
            M_logdet = logdet_grad_matrix(G_eval)
            @test M_logdet === G_eval.M_logdet  # Should be cached

            M_kinetic = kinetic_grad_matrix(G_eval, r)
            @test size(M_kinetic) == (D, D)

            # Test kinetic energy matches MvNormal logpdf
            G_matrix = G_eval.Q * Diagonal(G_eval.softabsλ) * G_eval.Q'
            @test neg_energy(hamiltonian, r, x) ≈
                logpdf(MvNormal(zeros(D), Symmetric(G_matrix)), r)

            # Test ∂H∂θ against finite differences
            Hamifunc = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
            Hamifuncx = x -> Hamifunc(x, r)
            @test δ(
                finite_difference_gradient(Hamifuncx, x), ∂H∂θ(hamiltonian, x, r).gradient
            ) < 1e-4

            # Test ∂H∂r against finite differences
            Hamifuncr = r -> Hamifunc(x, r)
            @test δ(finite_difference_gradient(Hamifuncr, r), ∂H∂r(hamiltonian, x, r)) <
                1e-4
        end
    end
end

####
#### Tests for deprecated API (DenseRiemannianMetric)
####

@testset "Deprecated DenseRiemannianMetric (backward compatibility)" begin
    @testset "$(nameof(typeof(target)))" for target in [HighDimGaussian(2), Funnel()]
        rng = MersenneTwister(1110)
        λ = 1e-2

        θ₀ = rand(rng, dim(target))

        ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
        ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, θ₀)

        Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample(ℓπ, θ₀, λ)

        D = dim(target)
        x = zeros(D)
        r = randn(rng, D)

        @testset "Autodiff utilities" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < 1e-4
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)[end]) < 1e-4
            @test δ(reshape_∂G∂θ(finite_difference_jacobian(Gfunc, x)), ∂G∂θfunc(x)) < 1e-4
        end

        @testset "$(nameof(typeof(hessmap)))" for hessmap in
                                                  [IdentityMap(), SoftAbsMap(20.0)]
            # Suppress deprecation warning
            metric = @test_deprecated DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, hessmap)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            if hessmap isa SoftAbsMap || all(iszero, x)
                @testset "Kinetic energy" begin
                    Σ = hamiltonian.metric.map(hamiltonian.metric.G(x))
                    @test neg_energy(hamiltonian, r, x) ≈ logpdf(MvNormal(zeros(D), Σ), r)
                end
            end

            Hamifunc = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
            Hamifuncx = x -> Hamifunc(x, r)
            Hamifuncr = r -> Hamifunc(x, r)

            @testset "∂H∂θ" begin
                @test δ(
                    finite_difference_gradient(Hamifuncx, x),
                    ∂H∂θ(hamiltonian, x, r).gradient,
                ) < 1e-4
            end

            @testset "∂H∂r" begin
                @test δ(finite_difference_gradient(Hamifuncr, r), ∂H∂r(hamiltonian, x, r)) <
                    1e-4
            end
        end
    end
end

####
#### Integration tests with sampling
####

@testset "Sampling with unified RiemannianMetric" begin
    n_samples = 100
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    metric = RiemannianMetric((D,), G, ∂G∂θ)
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=false)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

@testset "Sampling with SoftAbsRiemannianMetric" begin
    n_samples = 100
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    metric = SoftAbsRiemannianMetric((D,), G, ∂G∂θ, 20.0)
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=false)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

@testset "Sampling with deprecated DenseRiemannianMetric (IdentityMap)" begin
    n_samples = 100
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    metric = @test_deprecated DenseRiemannianMetric((D,), G, ∂G∂θ)
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=false)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

@testset "Sampling with deprecated DenseRiemannianMetric (SoftAbsMap)" begin
    n_samples = 100
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    metric = @test_deprecated DenseRiemannianMetric((D,), G, ∂G∂θ, SoftAbsMap(20.0))
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=false)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

####
#### Energy conservation tests
####

@testset "Energy conservation" begin
    rng = MersenneTwister(42)
    D_test = 2
    target = HighDimGaussian(D_test)
    θ₀ = rand(rng, D_test)
    λ = 1e-2

    ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
    ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, θ₀)
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, θ₀, λ)

    @testset "SoftAbsRiemannianMetric energy conservation" begin
        metric = SoftAbsRiemannianMetric((D_test,), G, ∂G∂θ, 20.0)
        kinetic = GaussianKinetic()
        hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

        # Small step size for better energy conservation
        integrator = GeneralizedLeapfrog(0.001, 10)

        # Create initial phase point
        θ_init = zeros(D_test)
        r_init = randn(rng, D_test)
        z0 = AdvancedHMC.phasepoint(hamiltonian, θ_init, r_init)
        H0 = -AdvancedHMC.neg_energy(z0)

        # Take 10 leapfrog steps
        z1 = AdvancedHMC.step(integrator, hamiltonian, z0, 10)
        H1 = -AdvancedHMC.neg_energy(z1)

        # Energy should be approximately conserved
        @test abs(H1 - H0) < 0.1
    end
end

####
#### Validation tests
####

@testset "Validation testing" begin
    target = HighDimGaussian(2)
    rng = MersenneTwister(125)
    λ = 1e-2

    initial_θ = rand(rng, dim(target))

    ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
    ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, initial_θ)

    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    D = dim(target)
    x = zeros(D)
    r = randn(rng, D)

    n_samples = 100
    n_adapts = 50

    mean_tol = 3 / sqrt(n_samples)
    var_tol = 1.5 * sqrt(2 / (n_samples - 1))

    @testset "RiemannianMetric (PDMat-style)" begin
        metric = RiemannianMetric((D,), G, ∂G∂θ)
        kinetic = GaussianKinetic()
        hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

        initial_ϵ = 0.01
        integrator = GeneralizedLeapfrog(initial_ϵ, 12)
        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

        acceptance_rate = 0.7
        adaptor = StepSizeAdaptor(acceptance_rate, integrator)

        samples, stats = sample(
            rng,
            hamiltonian,
            kernel,
            initial_θ,
            n_samples,
            adaptor,
            n_adapts;
            progress=false,
        )
        @test mean(samples) ≈ zeros(D) atol = mean_tol
        @test Statistics.var(samples) ≈ ones(D) atol = var_tol
    end

    @testset "SoftAbsRiemannianMetric" begin
        # We do not need SoftAbs for Gaussian target, so using small α
        metric = SoftAbsRiemannianMetric((D,), G, ∂G∂θ, 1.0)
        kinetic = GaussianKinetic()
        hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

        initial_ϵ = 0.01
        integrator = GeneralizedLeapfrog(initial_ϵ, 12)
        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

        acceptance_rate = 0.7
        adaptor = StepSizeAdaptor(acceptance_rate, integrator)

        samples, stats = sample(
            rng,
            hamiltonian,
            kernel,
            initial_θ,
            n_samples,
            adaptor,
            n_adapts;
            progress=false,
        )
        @test mean(samples) ≈ zeros(D) atol = mean_tol
        @test Statistics.var(samples) ≈ ones(D) atol = var_tol
    end
end
