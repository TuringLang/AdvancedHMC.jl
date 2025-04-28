using ReTest, Random
using AdvancedHMC, ForwardDiff, AbstractMCMC
using LinearAlgebra

using Pkg
Pkg.develop(; url="https://github.com/chalk-lab/MCMCLogDensityProblems.jl")
using MCMCLogDensityProblems

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
    _Hfunc = MCMCLogDensityProblems.gen_hess(x -> -ℓπ(x), initial_θ) # x -> (value, gradient, hessian)
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H + λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        all(isfinite, H) ? H : diagm(ones(length(x)))
    end
    _∂G∂θfunc = gen_∂G∂θ_fwd(x -> -ℓπ(x), initial_θ; f=fstabilize)
    ∂G∂θfunc = x -> reshape_∂G∂θ(_∂G∂θfunc(x))

    return Gfunc, ∂G∂θfunc
end

@testset "Multi variate Normal with Riemannian HMC" begin
    # Set the number of samples to draw and warmup iterations
    n_samples = 2_000
    rng = MersenneTwister(1110)
    initial_θ = rand(rng, D)
    λ = 1e-2
    G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)
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
    G, ∂G∂θ = prepare_sample(ℓπ, initial_θ, λ)

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
