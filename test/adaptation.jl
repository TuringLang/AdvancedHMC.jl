using Test, LinearAlgebra, Distributions, AdvancedHMC, Random, ForwardDiff
using AdvancedHMC.Adaptation: WelfordVar, NaiveVar, WelfordCov, NaiveCov, get_estimation, get_estimation, reset!

# Check that the estimated variance is approximately correct.
@testset "Online v.s. naive v.s. true var/cov estimation" begin
    D = 10
    T = Float64
    sz = (D,)
    n_samples = 100_000

    var_welford = WelfordVar{T}(sz)
    var_naive = NaiveVar{T}(sz)
    var_estimators = [var_welford, var_naive]
    cov_welford = WelfordCov{T}(sz)
    cov_naive = NaiveCov{T}(sz)
    cov_estimators = [cov_welford, cov_naive]
    estimators = [var_estimators..., cov_estimators...]

    for dist in [MvNormal(zeros(D), ones(D)), Dirichlet(D, 1)]
        for _ = 1:n_samples
            s = rand(dist)
            for estimator in estimators
                push!(estimator, s)
            end
        end

        @test get_estimation(var_welford) ≈ get_estimation(var_naive) atol=0.1D
        for estimator in var_estimators
            @test get_estimation(estimator) ≈ var(dist) atol=0.1D
        end

        @test get_estimation(cov_welford) ≈ get_estimation(cov_naive) atol=0.1D^2
        for estimator in cov_estimators
            @test get_estimation(estimator) ≈ cov(dist) atol=0.1D^2
        end

        for estimator in estimators
            reset!(estimator)
        end
    end
end

@testset "MassMatrixAdaptor constructors" begin
    θ = [0.0, 0.0, 0.0, 0.0]
    pc1 = MassMatrixAdaptor(UnitEuclideanMetric) # default dim = 2
    pc2 = MassMatrixAdaptor(DiagEuclideanMetric)
    pc3 = MassMatrixAdaptor(DenseEuclideanMetric)

    # Var adaptor dimention should be increased to length(θ) from 2
    AdvancedHMC.adapt!(pc1, θ, 1.)
    AdvancedHMC.adapt!(pc2, θ, 1.)
    AdvancedHMC.adapt!(pc3, θ, 1.)
    @test AdvancedHMC.Adaptation.getM⁻¹(pc2) == ones(length(θ))
    @test AdvancedHMC.Adaptation.getM⁻¹(pc3) == LinearAlgebra.diagm(0 => ones(length(θ)))
end

@testset "Stan HMC adaptors" begin
    θ = [0.0, 0.0, 0.0, 0.0]

    adaptor1 = StanHMCAdaptor(
        MassMatrixAdaptor(UnitEuclideanMetric),
        NesterovDualAveraging(0.8, 0.5)
    )
    adaptor2 = StanHMCAdaptor(
        MassMatrixAdaptor(DiagEuclideanMetric),
        NesterovDualAveraging(0.8, 0.5)
    )
    adaptor3 = StanHMCAdaptor(
        MassMatrixAdaptor(DenseEuclideanMetric),
        NesterovDualAveraging(0.8, 0.5)
    )
    for a in [adaptor1, adaptor2, adaptor3]
        AdvancedHMC.initialize!(a, 1_000)
        @test a.state.window_start == 76
        @test a.state.window_end == 950
        @test a.state.window_splits == [100, 150, 250, 450, 950]
        AdvancedHMC.adapt!(a, θ, 1.)
    end
    @test AdvancedHMC.Adaptation.getM⁻¹(adaptor2) == ones(length(θ))
    @test AdvancedHMC.Adaptation.getM⁻¹(adaptor3) == LinearAlgebra.diagm(0 => ones(length(θ)))

    @test_deprecated StanHMCAdaptor(
        1_000,
        MassMatrixAdaptor(DiagEuclideanMetric),
        NesterovDualAveraging(0.8, 0.5)
    )

    @testset "buffer > `n_adapts`" begin
        AdvancedHMC.initialize!(
            StanHMCAdaptor(
                MassMatrixAdaptor(DenseEuclideanMetric),
                NesterovDualAveraging(0.8, 0.5)
            ),
            100
        )
    end
end

let D=10
    function runnuts(ℓπ, metric; n_samples=3_000)
        n_adapts = 1_500

        θ_init = rand(D)

        h = Hamiltonian(metric, ℓπ, ForwardDiff)
        κ = NUTS(Leapfrog(find_good_stepsize(h, θ_init)))
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric),
            StepSizeAdaptor(0.8, κ.τ.integrator)
        )
        samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false)
        return (samples=samples, stats=stats, adaptor=adaptor)
    end

    @testset "Adapted mass v.s. true variance" begin
        n_tests = 5

        @testset "DiagEuclideanMetric" begin
            for _ in 1:n_tests
                Random.seed!(1)

                # Random variance
                σ = ones(D) + abs.(randn(D))

                # Diagonal Gaussian
                target = MvNormal(zeros(D), σ)
                ℓπ = θ -> logpdf(target, θ)

                res = runnuts(ℓπ, DiagEuclideanMetric(D))
                @test res.adaptor.pc.var ≈ σ .^ 2 rtol=0.2

                res = runnuts(ℓπ, DenseEuclideanMetric(D))
                @test res.adaptor.pc.cov ≈ diagm(0 => σ .^ 2) rtol=0.25
            end
        end

        @testset "DenseEuclideanMetric" begin
            for _ in 1:n_tests
                # Random covariance
                m = randn(D, D)
                Σ = m' * m

                # Correlated Gaussian
                target = MvNormal(zeros(D), Σ)
                ℓπ = θ -> logpdf(target, θ)

                res = runnuts(ℓπ, DiagEuclideanMetric(D))
                @test res.adaptor.pc.var ≈ diag(Σ) rtol=0.2

                res = runnuts(ℓπ, DenseEuclideanMetric(D))
                @test res.adaptor.pc.cov ≈ Σ rtol=0.25
            end
        end

    end

    @testset "Initialisation adaptor by metric" begin
        target = MvNormal(zeros(D), 1)
        ℓπ = θ -> logpdf(target, θ)

        mass_init = fill(0.5, D)
        res = runnuts(ℓπ, DiagEuclideanMetric(mass_init); n_samples=1)
        @test res.adaptor.pc.var == mass_init

        mass_init = diagm(0 => fill(0.5, D))
        res = runnuts(ℓπ, DenseEuclideanMetric(mass_init); n_samples=1)
        @test res.adaptor.pc.cov == mass_init
    end
end

@testset "Deprecation" begin
    dim = 10
    @test_deprecated Preconditioner(UnitEuclideanMetric(dim))
    @test_deprecated Preconditioner(DiagEuclideanMetric(dim))
    @test_deprecated Preconditioner(DenseEuclideanMetric(dim))
    @test_deprecated Preconditioner(UnitEuclideanMetric)
    @test_deprecated Preconditioner(DiagEuclideanMetric)
    @test_deprecated Preconditioner(DenseEuclideanMetric)
    for T in [Float32, Float64]
        @test_deprecated Preconditioner(T, UnitEuclideanMetric)
        @test_deprecated Preconditioner(T, DiagEuclideanMetric)
        @test_deprecated Preconditioner(T, DenseEuclideanMetric)
    end
    @test_deprecated NesterovDualAveraging(0.8, Leapfrog(0.1))
    @test_deprecated StanHMCAdaptor(100, MassMatrixAdaptor(UnitEuclideanMetric(dim)), StepSizeAdaptor(0.8, Leapfrog(0.1)))
end
