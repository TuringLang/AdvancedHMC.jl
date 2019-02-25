using Test, HMC
using Statistics: mean, var, cov
include("common.jl")

@testset "HMC" begin
    θ_init = randn(D)
    ϵ = 0.02
    n_steps = 20
    n_samples = 50_000

    p = TakeLastProposal(StaticTrajectory(Leapfrog(ϵ), n_steps))

    temp = randn(D,100)
    for metric in [UnitEuclideanMetric(θ_init), DiagEuclideanMetric(θ_init, vec(var(temp; dims=2))), DenseEuclideanMetric(θ_init, cov(temp'))]
        h = Hamiltonian(metric, _logπ, _dlogπdθ)

        @time samples = HMC.sample(h, p, θ_init, n_samples)

        @test mean(samples) ≈ zeros(D) atol=RNDATOL
    end
end
