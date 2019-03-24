using AdvancedHMC.Adaptation: WelfordVar, NaiveVar, WelfordCovar, NaiveCovar, add_sample!, get_var, get_covar, reset!
using Test, LinearAlgebra, Distributions

# Check that the estimated variance is approximately correct.
@testset "Online v.s. naive" begin
    D = 10
    n_samples = 10_000

    var_welford = WelfordVar(D)
    var_naive = NaiveVar()
    cov_welford = WelfordCovar(D)
    cov_naive = NaiveCovar()

    for dist in [MvNormal(zeros(D), ones(D))]
        for _ = 1:n_samples
            s = rand(dist)
            add_sample!(var_welford, s)
            add_sample!(var_naive, s)
            add_sample!(cov_welford, s)
            add_sample!(cov_naive, s)
        end

        @test get_var(var_welford) ≈ get_var(var_naive) atol=0.2
        @test get_covar(cov_welford) ≈ get_covar(cov_naive) atol=0.2

        reset!(var_welford)
        reset!(var_naive)
        reset!(cov_welford)
        reset!(cov_naive)
    end
end
