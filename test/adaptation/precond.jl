using AdvancedHMC.Adaptation: WelfordVar, NaiveVar, WelfordCov, NaiveCov, add_sample!, get_var, get_cov, reset!
using Test, LinearAlgebra, Distributions

# Check that the estimated variance is approximately correct.
@testset "Online v.s. naive v.s. true var/cov estimation" begin
    D = 10
    n_samples = 100_000

    var_welford = WelfordVar(D)
    var_naive = NaiveVar()
    var_estimators = [var_welford, var_naive]
    cov_welford = WelfordCov(D)
    cov_naive = NaiveCov()
    cov_estimators = [cov_welford, cov_naive]
    estimators = [var_estimators..., cov_estimators...]

    for dist in [MvNormal(zeros(D), ones(D)), Dirichlet(D, 1)]
        for _ = 1:n_samples
            s = rand(dist)
            for estimator in estimators
                add_sample!(estimator, s)
            end
        end

        @test get_var(var_welford) ≈ get_var(var_naive) atol=0.1D
        for estimator in var_estimators
            @test get_var(estimator) ≈ var(dist) atol=0.1D
        end

        @test get_cov(cov_welford) ≈ get_cov(cov_naive) atol=0.1D^2
        for estimator in cov_estimators
            @test get_cov(estimator) ≈ cov(dist) atol=0.1D^2
        end

        for estimator in estimators
            reset!(estimator)
        end
    end
end
