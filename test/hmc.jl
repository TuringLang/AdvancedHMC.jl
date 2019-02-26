using Test, HMC
using Statistics: mean, var, cov
include("common.jl")

@testset "HMC and NUTS" begin
    θ_init = randn(D)
    ϵ = 0.02
    n_steps = 20
    n_samples = 50_000

    temp = randn(D,100)
    for metric in [UnitEuclideanMetric(θ_init), DiagEuclideanMetric(θ_init, vec(var(temp; dims=2))), DenseEuclideanMetric(θ_init, cov(temp'))]
        h = Hamiltonian(metric, logπ, dlogπdθ)
        for p in [TakeLastProposal(StaticTrajectory(Leapfrog(ϵ), n_steps)), TakeLastProposal(NoUTurnTrajectory(Leapfrog(find_good_eps(h, θ_init))))]
            @time samples = HMC.sample(h, p, θ_init, n_samples)
            @test mean(samples) ≈ zeros(D) atol=RNDATOL
        end
    end
end
