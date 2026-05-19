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
    SoftAbsRiemannianMetric,
    make_J,
    _xcothx,
    _xcothx_taylor,
    _xcothx_exact,
    _xcothx_deriv,
    _xcothx_deriv_taylor,
    _xcothx_deriv_exact,
    _xcothx_cutoff
using Statistics

####
#### Test utilities
####

function gen_hess_fwd(func, x::AbstractVector)
    cfg = ForwardDiff.HessianConfig(func, x)
    H = Matrix{eltype(x)}(undef, length(x), length(x))

    function hess(x::AbstractVector)
        ForwardDiff.hessian!(H, func, x, cfg)
        return H
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    chunk = ForwardDiff.Chunk(x)
    tag = ForwardDiff.Tag(Vfunc, eltype(x))
    jac_cfg = ForwardDiff.JacobianConfig(Vfunc, x, chunk, tag)
    hess_cfg = ForwardDiff.HessianConfig(Vfunc, jac_cfg.duals, chunk, tag)

    d = length(x)
    out = zeros(eltype(x), d^2, d)

    function ∂G∂θ_fwd(y)
        hess = z -> ForwardDiff.hessian(Vfunc, z, hess_cfg, Val{false}())
        ForwardDiff.jacobian!(out, hess, y, jac_cfg, Val{false}())
        return out
    end

    return ∂G∂θ_fwd
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[((i - 1) * d + 1):(i * d), :] for i in 1:d)...; dims=3)
end

function prepare_sample(ℓπ, initial_θ, λ)
    Vfunc = x -> -ℓπ(x)
    _Hfunc = gen_hess_fwd(Vfunc, initial_θ)
    Hfunc = x -> copy.(_Hfunc(x))

    fstabilize = H -> H + λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x))
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(x -> -ℓπ(x), initial_θ; f=fstabilize)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x))

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

δ(a, b) = maximum(abs.(a - b))

####
#### Tests for SoftAbs numerical-stability helpers (P0.1)
####

@testset "SoftAbs stability helpers" begin
    @testset "no NaN at x = 0 (previously Inf − Inf)" begin
        # Limits: x·coth(x) → 1, its derivative → 0 as x → 0.
        @test _xcothx(0.0) == 1.0
        @test _xcothx_deriv(0.0) == 0.0
    end

    @testset "Taylor and exact branches agree at the switch threshold" begin
        # Compare the two branch implementations directly at the threshold.
        # Tolerances are conservative — the actual disagreement is bounded by
        # the larger of Taylor truncation (~eps at threshold for both) and exact-
        # form cancellation (~6e-14 for _xcothx_deriv).
        thresh = _xcothx_cutoff(Float64)
        @test _xcothx_taylor(thresh) ≈ _xcothx_exact(thresh) atol = 1e-12
        @test _xcothx_deriv_taylor(thresh) ≈ _xcothx_deriv_exact(thresh) atol = 1e-10
    end

    @testset "make_J: no NaN at λ = 0 and matches exact formula" begin
        α = 20.0
        # Mixed: one zero, one well-separated, one negative — the previously-broken case.
        λ = [0.0, 1.7, -0.5]
        J = make_J(λ, α)
        @test all(isfinite, J)
        @test J ≈ J'
        # Off-diagonal entries not touching λ=0 match the analytic divided difference of
        # softabs(λ) = λ·coth(αλ).
        @test J[2, 3] ≈
            (λ[2] * coth(α * λ[2]) - λ[3] * coth(α * λ[3])) / (λ[2] - λ[3]) rtol = 1e-10
        # Diagonal at λ=0 is the well-defined limit softabs'(0) = 0.
        @test J[1, 1] == 0.0
        # Diagonal at λ≠0 matches softabs'(λ) = coth(αλ) − αλ·csch²(αλ).
        @test J[2, 2] ≈ coth(α * λ[2]) - α * λ[2] * csch(α * λ[2])^2 rtol = 1e-10
    end
end

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
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)) < 1e-4
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
    # 1D Wasserstein-1 distance
    function w1(a::AbstractVector, b::AbstractVector)
        sa = sort(a)
        sb = sort(b)
        return mean(abs.(sa .- sb))
    end

    @testset "Validation testing (Gaussian)" begin

        # 1D normal Wasserstein-1 distance tolerance estimator
        function w1_tol_normal_1d(;
            n::Int, reps::Int=200, q::Float64=0.999, rng=Random.default_rng()
        )
            poolN = max(50_000, 50n)
            pool = randn(rng, poolN)

            vals = Vector{Float64}(undef, reps)
            for i in 1:reps
                a = pool[rand(rng, 1:poolN, n)]
                b = pool[rand(rng, 1:poolN, n)]
                vals[i] = w1(a, b)
            end
            sort!(vals)
            return vals[clamp(ceil(Int, q * reps), 1, reps)]
        end

        target = HighDimGaussian(2)
        rng = MersenneTwister(125)
        λ = 1e-2

        initial_θ = rand(rng, dim(target))

        ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
        ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, initial_θ)

        _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

        D = dim(target)

        n_samples = 100
        n_adapts = 50

        tol_w1 = w1_tol_normal_1d(; n=n_samples, rng=rng)

        tol_w1 *= 1.5

        x_true = randn(rng, n_samples)
        y_true = randn(rng, n_samples)

        @testset "RiemannianMetric (PDMat-style)" begin
            metric = RiemannianMetric((D,), G, ∂G∂θ)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            initial_ϵ = 0.01
            integrator = GeneralizedLeapfrog(initial_ϵ, 15)
            kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

            acceptance_rate = 0.9
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
            θ = reduce(vcat, (permutedims(s) for s in samples))
            # 1st marginal
            @test w1(θ[:, 1], x_true) < tol_w1
            # 2nd marginal
            @test w1(θ[:, 2], y_true) < tol_w1
        end

        @testset "SoftAbsRiemannianMetric" begin
            # We do not need SoftAbs for Gaussian target, so using small α
            metric = SoftAbsRiemannianMetric((D,), G, ∂G∂θ, 1.0)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            initial_ϵ = 0.01
            integrator = GeneralizedLeapfrog(initial_ϵ, 15)
            kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

            acceptance_rate = 0.9
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

            θ = reduce(vcat, (permutedims(s) for s in samples))
            # 1st marginal
            @test w1(θ[:, 1], x_true) < tol_w1
            # 2nd marginal
            @test w1(θ[:, 2], y_true) < tol_w1
        end
    end

    @testset "Validation testing (Funnel)" begin

        # Funnel i.i.d. sampler
        # θ layout: [v, x1]
        function funnel_iid(rng::AbstractRNG, n::Int)
            v = 3.0 .* randn(rng, n)
            X = Matrix{Float64}(undef, n, 1)
            for i in 1:n
                s = exp(v[i] / 2)
                @inbounds X[i, :] .= s .* randn(rng, 1)
            end
            return v, X
        end

        # 1D Wasserstein-1 distance tolerances for Funnel marginals
        function funnel_w1_tols(;
            n::Int,
            reps::Int=200,
            q::Float64=0.999,
            inflate::Float64=1.8,
            rng::AbstractRNG=Random.default_rng(),
        )
            vals_v = Vector{Float64}(undef, reps)
            vals_x1 = Vector{Float64}(undef, reps)

            for i in 1:reps
                vA, XA = funnel_iid(rng, n)
                vB, XB = funnel_iid(rng, n)

                x1A = XA[:, 1]
                x1B = XB[:, 1]

                vals_v[i] = w1(vA, vB)
                vals_x1[i] = w1(x1A, x1B)
            end

            sort!(vals_v)
            sort!(vals_x1)
            idx = clamp(ceil(Int, q * reps), 1, reps)

            return (tol_v=inflate * vals_v[idx], tol_x1=inflate * vals_x1[idx])
        end

        target = Funnel()
        rng = MersenneTwister(234)
        λ = 1e-2

        initial_θ = rand(rng, dim(target))

        ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
        ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, initial_θ)

        _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

        D = dim(target)

        n_samples = 1000
        n_adapts = 500

        # True samples
        v_true, X_true = funnel_iid(rng, n_samples)

        # Wasserstein-1 distance tolerances
        tols = funnel_w1_tols(; n=n_samples, rng=rng)

        @testset "SoftAbsRiemannianMetric" begin
            metric = SoftAbsRiemannianMetric((D,), G, ∂G∂θ, 40.0)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            initial_ϵ = 0.01
            integrator = GeneralizedLeapfrog(initial_ϵ, 5)
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

            θ = reduce(vcat, (permutedims(s) for s in samples))
            # 1st marginal
            @test w1(θ[:, 1], v_true) < tols.tol_v
            # 2nd marginal
            @test w1(θ[:, 2], X_true[:, 1]) < tols.tol_x1
        end
    end
end
