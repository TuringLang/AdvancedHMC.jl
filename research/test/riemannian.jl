using ReTest, AdvancedHMC
using AdvancedHMC.Experimental
using AdvancedHMC: ∂H∂r, ∂H∂θ
using AdvancedHMC.Experimental: AbstractRelativisticKinetic

using FiniteDiff:
    finite_difference_gradient, finite_difference_hessian, finite_difference_jacobian
using Distributions: MvNormal
using AdvancedHMC: neg_energy, energy

const TARGET_TOL = 1e-4
const HAMILTONIAN_TOL = 1e-3
# Taken from https://github.com/JuliaDiff/FiniteDiff.jl/blob/master/test/finitedifftests.jl
δ(a, b) = maximum(abs.(a - b))

@testset "RiemannianHMC" begin

    hps = (; λ = 1e-2, α = 20.0, ϵ = 0.1, n = 6, L = 8, m = 1.5, c = 1.5)

    @testset "$(nameof(typeof(target)))" for target in [
        HighDimGaussian(2), 
        Funnel(),
    ]
        rng = MersenneTwister(1110)

        θ₀ = rand(rng, dim(target))

        ℓπ = VecTargets.gen_logpdf(target)
        ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)

        Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ)

        D = dim(target) # ==2 for this test
        x_zero = zeros(D)
        x_rand = randn(rng, D)
        r = randn(rng, D)

        @testset "Autodiff" begin
            x = x_rand
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < TARGET_TOL
            @test δ(finite_difference_hessian(Vfunc, x), Hfunc(x)[end]) < TARGET_TOL
            # finite_difference_jacobian returns shape of (4, 2), reshape_∂G∂θ turns it into (2, 2, 2)
            @test δ(reshape_∂G∂θ(finite_difference_jacobian(Gfunc, x)), ∂G∂θfunc(x)) < TARGET_TOL
        end

        @testset "$(nameof(typeof(hessmap)))" for hessmap in [
                IdentityMap(), 
                SoftAbsMap(hps.α),
            ]
            x = hessmap isa IdentityMap ? x_zero : x_rand
            metric = DenseRiemannianMetric((D,), Gfunc, ∂G∂θfunc, hessmap)
            @testset "$(nameof(typeof(kinetic)))" for kinetic in [
                    GaussianKinetic(), 
                    RelativisticKinetic(hps.m, hps.c), 
                    DimensionwiseRelativisticKinetic(hps.m, hps.c),
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
                
                # NOTE the implementation of ∂H∂θ for dimensionwise relativistic kinetic is not correct
                #      thus giving expected broken tests for Funnel and DimensionwiseRelativisticKinetic
                broken = target isa Funnel && kinetic isa DimensionwiseRelativisticKinetic

                if !(hessmap isa IdentityMap && kinetic isa AbstractRelativisticKinetic) # no plan to implement rel-hmc for Fisher metric
                    @testset "∂H∂θ" begin
                        @test δ(
                            finite_difference_gradient(Hamifuncx, x),
                            ∂H∂θ(hamiltonian, x, r).gradient,
                        ) < HAMILTONIAN_TOL broken=broken
                    end
                end

                @testset "∂H∂r" begin
                    @test δ(
                        finite_difference_gradient(Hamifuncr, r), 
                        ∂H∂r(hamiltonian, x, r),
                    ) < HAMILTONIAN_TOL
                end
            end
        end
    end
end
