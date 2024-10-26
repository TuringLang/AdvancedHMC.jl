using ReTest, AdvancedHMC
using AdvancedHMC.Experimental
using AdvancedHMC: ∂H∂r, ∂H∂θ

using LinearAlgebra: dot

@testset "RelativisticHMC" begin

    @testset "Hamiltonian" begin
        @testset "Construction" begin
            f = x -> dot(x, x)
            g = x -> 2x
            metric = UnitEuclideanMetric(10)
            h = Hamiltonian(metric, RelativisticKinetic(1.0, 1.0), f, g)
            @test h.kinetic isa RelativisticKinetic
        end
    end

    # TODO Convert the test below into a function `autodiff_test`
    hps = (; m = 1.5, c = 1.5)

    @testset "$(nameof(typeof(target)))" for target in [
        # HighDimGaussian(2), 
        Funnel(),
    ]
        rng = MersenneTwister(1110)

        θ₀ = rand(rng, dim(target))

        ℓπ = VecTargets.gen_logpdf(target)
        ∂ℓπ∂θ = VecTargets.gen_logpdf_grad(target, θ₀)

        Vfunc, Hfunc, Gfunc, ∂G∂θfunc = prepare_sample_target(hps, θ₀, ℓπ)

        D = dim(target) # ==2 for this test
        x = randn(rng, D)
        r = randn(rng, D)

        @testset "Autodiff" begin
            @test δ(finite_difference_gradient(ℓπ, x), ∂ℓπ∂θ(x)[end]) < TARGET_TOL
        end

        @testset "$(nameof(typeof(metric)))" for metric in [
                UnitEuclideanMetric(D),
                DiagEuclideanMetric(D),
                DiagEuclideanMetric([0.5, 4.0]),
                DenseEuclideanMetric(D),
                DenseEuclideanMetric([1.32 0.79; 0.79 0.48]),
            ]
            @testset "$(nameof(typeof(kinetic)))" for kinetic in [
                RelativisticKinetic(hps.m, hps.c), 
                DimensionwiseRelativisticKinetic(hps.m, hps.c),
            ]
                hamiltonian = Hamiltonian(metric, kinetic, ℓπ, ∂ℓπ∂θ)

                Hamifunc = (x, r) -> energy(hamiltonian, r, x) + energy(hamiltonian, x)
                Hamifuncx = x -> Hamifunc(x, r)
                Hamifuncr = r -> Hamifunc(x, r)

                @testset "∂H∂θ" begin
                    @test δ(
                        finite_difference_gradient(Hamifuncx, x),
                        ∂H∂θ(hamiltonian, x, r).gradient,
                    ) < HAMILTONIAN_TOL
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
