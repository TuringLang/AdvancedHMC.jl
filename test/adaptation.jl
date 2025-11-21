using ReTest, LinearAlgebra, Distributions, AdvancedHMC, Random, ForwardDiff
using AdvancedHMC:
    PhasePoint, DualValue
using AdvancedHMC.Adaptation:
    DiagMatrixEstimator, WelfordVar, NutpieVar, NaiveVar, WelfordCov, NaiveCov, get_estimation, get_estimation, reset!

function runnuts(ℓπ, metric; n_samples=10_000)
    D = size(metric, 1)
    n_adapts = 5_000
    θ_init = rand(D)
    rng = MersenneTwister(0)

    nuts = NUTS(0.8)
    h = Hamiltonian(metric, ℓπ, ForwardDiff)
    step_size = AdvancedHMC.make_step_size(rng, nuts, h, θ_init)
    integrator = AdvancedHMC.make_integrator(nuts, step_size)
    κ = AdvancedHMC.make_kernel(nuts, integrator)
    adaptor = AdvancedHMC.make_adaptor(nuts, metric, integrator)
    samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false)
    return (samples=samples, stats=stats, adaptor=adaptor)
end

# Temporary function until we've settled on a different interface
function runnuts_nutpie(ℓπ, metric::DiagEuclideanMetric; n_samples=10_000)
    D = size(metric, 1)
    n_adapts = 5_000
    θ_init = rand(D)
    rng = MersenneTwister(0)

    nuts = NUTS(0.8)
    h = Hamiltonian(metric, ℓπ, ForwardDiff)
    step_size = AdvancedHMC.make_step_size(rng, nuts, h, θ_init)
    integrator = AdvancedHMC.make_integrator(nuts, step_size)
    κ = AdvancedHMC.make_kernel(nuts, integrator)
    # Constructing like this until we've settled on a different interface
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.Adaptation.NutpieVar(size(metric); var=copy(metric.M⁻¹)), 
        AdvancedHMC.StepSizeAdaptor(nuts.δ, integrator)
    )
    samples, stats = sample(h, κ, θ_init, n_samples, adaptor, n_adapts; verbose=false)
    return (samples=samples, stats=stats, adaptor=adaptor)
end
preconditioned_cond(a::DiagMatrixEstimator, cov::AbstractMatrix) = cond(sqrt(Diagonal(a.var)) \ cov / sqrt(Diagonal(a.var)))

@testset "Adaptation" begin
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

        for dist in [MvNormal(zeros(D), I), Dirichlet(D, 1)]
            for _ in 1:n_samples
                s = rand(dist)
                for estimator in estimators
                    push!(estimator, s)
                end
            end

            @test get_estimation(var_welford) ≈ get_estimation(var_naive) atol = 0.1D
            for estimator in var_estimators
                @test get_estimation(estimator) ≈ var(dist) atol = 0.1D
            end

            @test get_estimation(cov_welford) ≈ get_estimation(cov_naive) atol = 0.1D^2
            for estimator in cov_estimators
                @test get_estimation(estimator) ≈ cov(dist) atol = 0.1D^2
            end

            for estimator in estimators
                reset!(estimator)
            end
        end
    end

    @testset "MassMatrixAdaptor constructors" begin
        θ = [0.0, 0.0, 0.0, 0.0]
        z = PhasePoint(
            θ, θ, DualValue(0., θ), DualValue(0., θ)
        )
        pc1 = MassMatrixAdaptor(UnitEuclideanMetric) # default dim = 2
        pc2 = MassMatrixAdaptor(DiagEuclideanMetric)
        # Constructing like this until we've settled on a different interface
        pc2_nutpie = NutpieVar{Float64}((2, ))
        pc3 = MassMatrixAdaptor(DenseEuclideanMetric)

        # Var adaptor dimension should be increased to length(θ) from 2
        AdvancedHMC.adapt!(pc1, θ, 1.0)
        AdvancedHMC.adapt!(pc2, θ, 1.0)
        AdvancedHMC.adapt!(pc2_nutpie, z, 1.0)
        AdvancedHMC.adapt!(pc3, θ, 1.0)
        @test AdvancedHMC.Adaptation.getM⁻¹(pc2) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(pc2_nutpie) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(pc3) ==
            LinearAlgebra.diagm(0 => ones(length(θ)))

        # Making sure "all" MassMatrixAdaptors support getting a PhasePoint instead of a Vector
        # AdvancedHMC.adapt!(pc1, z, 1.0) - this does not work - should it?
        AdvancedHMC.adapt!(pc2, z, 1.0)
        AdvancedHMC.adapt!(pc2_nutpie, z, 1.0)
        AdvancedHMC.adapt!(pc3, z, 1.0)
        @test AdvancedHMC.Adaptation.getM⁻¹(pc2) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(pc2_nutpie) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(pc3) ==
            LinearAlgebra.diagm(0 => ones(length(θ)))
    end

    @testset "Stan HMC adaptors" begin
        θ = [0.0, 0.0, 0.0, 0.0]

        adaptor1 = StanHMCAdaptor(
            MassMatrixAdaptor(UnitEuclideanMetric), NesterovDualAveraging(0.8, 0.5)
        )
        adaptor2 = StanHMCAdaptor(
            MassMatrixAdaptor(DiagEuclideanMetric), NesterovDualAveraging(0.8, 0.5)
        )
        # Constructing like this until we've settled on a different interface
        adaptor2_nutpie = StanHMCAdaptor(
            NutpieVar{Float64}((2, )), NesterovDualAveraging(0.8, 0.5)
        )
        adaptor3 = StanHMCAdaptor(
            MassMatrixAdaptor(DenseEuclideanMetric), NesterovDualAveraging(0.8, 0.5)
        )
        for a in [adaptor1, adaptor2, adaptor2_nutpie, adaptor3]
            AdvancedHMC.initialize!(a, 1_000)
            @test a.state.window_start == 76
            @test a.state.window_end == 950
            @test a.state.window_splits == [100, 150, 250, 450, 950]
            AdvancedHMC.adapt!(a, θ, 1.0)
        end
        @test AdvancedHMC.Adaptation.getM⁻¹(adaptor2) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(adaptor2a) == ones(length(θ))
        @test AdvancedHMC.Adaptation.getM⁻¹(adaptor3) ==
            LinearAlgebra.diagm(0 => ones(length(θ)))

        @test_deprecated StanHMCAdaptor(
            1_000, MassMatrixAdaptor(DiagEuclideanMetric), NesterovDualAveraging(0.8, 0.5)
        )

        @testset "buffer > `n_adapts`" begin
            AdvancedHMC.initialize!(
                StanHMCAdaptor(
                    MassMatrixAdaptor(DenseEuclideanMetric), NesterovDualAveraging(0.8, 0.5)
                ),
                100,
            )
        end
    end

    @testset "Adapted mass v.s. true variance" begin
        D = 10
        n_tests = 5
        @testset "DiagEuclideanMetric" begin
            for _ in 1:n_tests
                Random.seed!(1)

                # Random variance
                σ² = 1 .+ abs.(randn(D))
                Σ = Diagonal(σ²)

                # Diagonal Gaussian
                ℓπ = LogDensityDistribution(MvNormal(Σ))

                res = runnuts(ℓπ, DiagEuclideanMetric(D))
                @test res.adaptor.pc.var ≈ σ² rtol = 0.2

                # For this target, Nutpie (without regularization) will arrive at the true variances after two draws.
                res_nutpie = runnuts_nutpie(ℓπ, DiagEuclideanMetric(D))
                @test res.adaptor.pc.var ≈ σ² rtol = 0.2
                @test preconditioned_cond(res_nutpie.adaptor.pc, Σ) < preconditioned_cond(res.adaptor.pc, Σ)

                res = runnuts(ℓπ, DenseEuclideanMetric(D))
                @test res.adaptor.pc.cov ≈ Diagonal(σ²) rtol = 0.25
            end
        end

        @testset "DenseEuclideanMetric" begin
            n_nutpie_superior = 0
            for _ in 1:n_tests
                # Random covariance
                m = randn(D, D)
                Σ = m' * m

                # Correlated Gaussian
                ℓπ = LogDensityDistribution(MvNormal(Σ))

                res = runnuts(ℓπ, DiagEuclideanMetric(D))
                @test res.adaptor.pc.var ≈ diag(Σ) rtol = 0.2

                # For this target, Nutpie will NOT converge towards the true variances, even after infinite draws.
                # HOWEVER, it will asymptotically (but also generally more quickly than Stan) 
                # find the best preconditioner for the target.
                # As these are statistical algorithms, superiority is not always guaranteed, hence this way of testing.  
                res_nutpie = runnuts_nutpie(ℓπ, DiagEuclideanMetric(D))
                # @test res.adaptor.pc.var ≈ diag(Σ) rtol = 0.2
                n_nutpie_superior += preconditioned_cond(res_nutpie.adaptor.pc, Σ) < preconditioned_cond(res.adaptor.pc, Σ)

                res = runnuts(ℓπ, DenseEuclideanMetric(D))
                @test res.adaptor.pc.cov ≈ Σ rtol = 0.25
            end
            @test n_nutpie_superior > n_tests / 2
        end
    end

    @testset "Initialisation adaptor by metric" begin
        ℓπ = LogDensityDistribution(MvNormal(Eye(D)))

        mass_init = fill(0.5, D)
        res = runnuts(ℓπ, DiagEuclideanMetric(mass_init); n_samples=1)
        @test res.adaptor.pc.var == mass_init

        mass_init = fill(0.5, D)
        res = runnuts_nutpie(ℓπ, DiagEuclideanMetric(mass_init); n_samples=1)
        @test res.adaptor.pc.var == mass_init

        mass_init = diagm(0 => fill(0.5, D))
        res = runnuts(ℓπ, DenseEuclideanMetric(mass_init); n_samples=1)
        @test res.adaptor.pc.cov == mass_init
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
        @test_deprecated StanHMCAdaptor(
            100,
            MassMatrixAdaptor(UnitEuclideanMetric(dim)),
            StepSizeAdaptor(0.8, Leapfrog(0.1)),
        )
    end
end
