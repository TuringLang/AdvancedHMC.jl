using ReTest, AdvancedHMC

include("../src/riemannian_hmc.jl")
include("../src/riemannian_hmc-sampler.jl")

using FiniteDiff: finite_difference_gradient, finite_difference_hessian, finite_difference_jacobian
using Distributions: MvNormal
using AdvancedHMC: neg_energy, energy

# Taken from https://github.com/JuliaDiff/FiniteDiff.jl/blob/master/test/finitedifftests.jl
δ(a, b) = maximum(abs.(a - b))

@testset "Riemannian" begin

    hps = (; λ=1e-2, α=20.0, ϵ=0.1, n=6, L=8)

    @testset "$(nameof(typeof(target)))" for target in [HighDimGaussian(2), Funnel()]
        rng = MersenneTwister(1110)

        D = dim(target) # ==2 for this test
        initial_θ = rand(rng, D)

        ℓπ = x -> logpdf(target, x)
        neg_ℓπ = x -> -logpdf(target, x)

        _∂ℓπ∂θ = gen_logpdf_grad(target, initial_θ)
        ∂ℓπ∂θ = x -> copy.(_∂ℓπ∂θ(x))
        
        _hess_func = VecTargets.gen_hess(neg_ℓπ, initial_θ) # x -> (value, gradient, hessian)
        hess_func = x -> copy.(_hess_func(x))

        G = x -> begin
            H = hess_func(x)[3] + hps.λ * I
            any(.!(isfinite.(H))) ? diagm(ones(length(x))) : H
        end
        _∂G∂θ = gen_∂H∂x(neg_ℓπ, initial_θ) # size==(4, 2)
        ∂G∂θ = x -> reshape_∂H∂x(copy(_∂G∂θ(x))) # size==(2, 2, 2)

        x = randn(rng, 2)
        r = randn(rng, 2)

        @testset "Autodiff" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < 1e-4
            @test δ(finite_difference_hessian(neg_ℓπ, x), hess_func(x)[end]) < 1e-4
            # finite_difference_jacobian returns shape of (4, 2)
            @test δ(finite_difference_jacobian(G, x), _∂G∂θ(x)) < 1e-4
        end

        @testset "$(nameof(typeof(hessmap)))" for hessmap in [IdentityMap(), SoftAbsMap(hps.α)]
            metric = DenseRiemannianMetric((D,), G, ∂G∂θ, hessmap)
            kinetic = GaussianKinetic()
            hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

            if hessmap isa SoftAbsMap # only test kinetic energy for SoftAbsMap as that of IdentityMap can be non-PD
                @testset "Kinetic energy" begin
                    Σ = hamiltonian.metric.map(hamiltonian.metric.G(x))
                    @test neg_energy(hamiltonian, r, x) ≈ logpdf(MvNormal(zeros(D), Σ), r)
                end
            end

            H_func = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
            Hx = x -> H_func(x, r)
            Hr = r -> H_func(x, r)

            @testset "∂H∂θ" begin
                @test δ(finite_difference_gradient(Hx, x), ∂H∂θ(hamiltonian, x, r).gradient) < 1e-4
            end

            @testset "∂H∂r" begin
                @test δ(finite_difference_gradient(Hr, r), ∂H∂r(hamiltonian, x, r)) < 1e-4
            end
        end

    end

end
