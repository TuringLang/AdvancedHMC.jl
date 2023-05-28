using ReTest, AdvancedHMC

include("../src/riemannian_hmc_utility.jl")

using FiniteDiff:
    finite_difference_gradient, finite_difference_hessian, finite_difference_jacobian
using Distributions: MvNormal
using AdvancedHMC: neg_energy, energy

# Taken from https://github.com/JuliaDiff/FiniteDiff.jl/blob/master/test/finitedifftests.jl
δ(a, b) = maximum(abs.(a - b))

const TARGET_TOL = 1e-4
const HAMILTONIAN_TOL = 2e-3

@testset "Riemannian" begin

    hps = (; λ = 1e-2, α = 20.0, ϵ = 0.1, n = 6, L = 8)

    @testset "$(nameof(typeof(target)))" for target in [HighDimGaussian(2), Funnel()]
        rng = MersenneTwister(1110)

        θ₀ = rand(rng, dim(target))

        ℓπ = VecTargets.gen_logpdf(target)
        ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)

        Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ)

        D = dim(target) # ==2 for this test
        x = zeros(D) # randn(rng, D)
        r = randn(rng, D)

        @testset "Autodiff" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < TARGET_TOL
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)[end]) < TARGET_TOL
            # finite_difference_jacobian returns shape of (4, 2), reshape_∂G∂θ turns it into (2, 2, 2)
            @test δ(reshape_∂G∂θ(finite_difference_jacobian(Gfunc, x)), ∂G∂θfunc(x)) < TARGET_TOL
        end

        @testset "$(nameof(typeof(hessmap)))" for hessmap in [
                IdentityMap(), 
                SoftAbsMap(hps.α),
            ]
            metric = DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, hessmap)
            @testset "$(nameof(typeof(kinetic)))" for kinetic in [
                    GaussianKinetic(), 
                    RelativisticKinetic(1.0, 1.0), 
                    DimensionwiseRelativisticKinetic(1.0, 1.0),
                ]
                hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

                if kinetic isa GaussianKinetic && # only test Gaussian kinetic energy for SoftAbsMap 
                    (hessmap isa SoftAbsMap || all(iszero.(x))) # that of IdentityMap can be non-PD for x==0 that I know it's PD
                    @testset "Kinetic energy" begin
                        Σ = hamiltonian.metric.map(hamiltonian.metric.G(x))
                        @test neg_energy(hamiltonian, r, x) ≈ logpdf(MvNormal(zeros(D), Σ), r)
                    end
                end

                Hamifunc = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
                Hamifuncx = x -> Hamifunc(x, r)
                Hamifuncr = r -> Hamifunc(x, r)

                if !(hessmap isa IdentityMap && kinetic isa AbstractRelativisticKinetic) # no plan to implement rel-hmc for Fisher metric
                    @testset "∂H∂θ" begin
                        @test δ(
                            finite_difference_gradient(Hamifuncx, x),
                            ∂H∂θ(hamiltonian, x, r).gradient,
                        ) < HAMILTONIAN_TOL
                    end
                end

                @testset "∂H∂r" begin
                    @test δ(finite_difference_gradient(Hamifuncr, r), ∂H∂r(hamiltonian, x, r)) <
                          HAMILTONIAN_TOL
                end
            end
        end
    end
end
