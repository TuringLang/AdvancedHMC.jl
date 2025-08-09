using ReTest, Random
using AdvancedHMC, ForwardDiff, AbstractMCMC
using LinearAlgebra
using MCMCLogDensityProblems
using FiniteDiff:
    finite_difference_gradient, finite_difference_hessian, finite_difference_jacobian
using AdvancedHMC: neg_energy, energy, ∂H∂θ, ∂H∂r

# Fisher information metric
function gen_∂G∂θ_fwd(Vfunc, x; f=identity)
    _Hfunc = gen_hess_fwd(Vfunc, x)
    Hfunc = x -> _Hfunc(x)[3]
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
    return out # default output shape [∂H∂x₁; ∂H∂x₂; ...]
end

function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[((i - 1) * d + 1):(i * d), :] for i in 1:d)...; dims=3)
end

function prepare_sample(ℓπ, initial_θ, λ)
    Vfunc = x -> -ℓπ(x)
    _Hfunc = MCMCLogDensityProblems.gen_hess(Vfunc, initial_θ) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H + λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(x -> -ℓπ(x), initial_θ; f=fstabilize)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x))

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end

@testset "Constructors tests" begin
    δ(a, b) = maximum(abs.(a - b))
    @testset "$(nameof(typeof(target)))" for target in [HighDimGaussian(2), Funnel()]
        rng = MersenneTwister(1110)
        λ = 1e-2

        θ₀ = rand(rng, dim(target))

        ℓπ = MCMCLogDensityProblems.gen_logpdf(target)
        ∂ℓπ∂θ = MCMCLogDensityProblems.gen_logpdf_grad(target, θ₀)

        Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample(ℓπ, θ₀, λ)

        D = dim(target) # ==2 for this test
        x = zeros(D) # randn(rng, D)
        r = randn(rng, D)

        @testset "Autodiff" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < 1e-4
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)[end]) < 1e-4
            # finite_difference_jacobian returns shape of (4, 2), reshape_∂G∂θ turns it into (2, 2, 2)
            @test δ(reshape_∂G∂θ(finite_difference_jacobian(Gfunc, x)), ∂G∂θfunc(x)) < 1e-4
        end

        @testset "$(nameof(typeof(hessmap)))" for hessmap in
                                                  [IdentityMap(), SoftAbsMap(20.0)]
            metric = DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, hessmap)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            if hessmap isa SoftAbsMap || # only test kinetic energy for SoftAbsMap as that of IdentityMap can be non-PD
                all(iszero, x) # or for x==0 that I know it's PD
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

@testset "Multi variate Normal with Riemannian HMC" begin
    # Set the number of samples to draw and warmup iterations
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)
    # Define a Hamiltonian system
    metric = DenseRiemannianMetric((D,), G, ∂G∂θ)
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)

    # Define an HMC sampler with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=true)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end

@testset "Multi variate Normal with Riemannian HMC softabs metric" begin
    # Set the number of samples to draw and warmup iterations
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    _, _, G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

    # Define a Hamiltonian system
    metric = DenseRiemannianMetric((D,), G, ∂G∂θ, λSoftAbsMap(20.0))
    kinetic = GaussianKinetic()
    hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = 0.01
    integrator = GeneralizedLeapfrog(initial_ϵ, 6)

    # Define an HMC sampler with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(8)))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(rng, hamiltonian, kernel, initial_θ, n_samples; progress=true)
    @test length(samples) == n_samples
    @test length(stats) == n_samples
end
