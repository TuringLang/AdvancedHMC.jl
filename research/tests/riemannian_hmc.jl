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

        _, ℓπ, ∂ℓπ∂θ, Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample_target(rng, hps, target)

        D = dim(target) # ==2 for this test
        x = zeros(D) # randn(rng, D)
        r = randn(rng, D)

        @testset "Autodiff" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < 1e-4
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)[end]) < 1e-4
            # finite_difference_jacobian returns shape of (4, 2), reshape_∂G∂x turns it into (2, 2, 2)
            @test δ(reshape_∂G∂x(finite_difference_jacobian(Gfunc, x)), ∂G∂θfunc(x)) < 1e-4
        end
        
        @testset "$(nameof(typeof(hessmap)))" for hessmap in [IdentityMap(), SoftAbsMap(hps.α)]
            metric = DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, hessmap)
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
