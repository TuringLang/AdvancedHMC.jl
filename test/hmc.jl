using Test, HMC
using Statistics: mean, var, cov
include("common.jl")

@testset "HMC" begin
    temp = randn(D,100)
    for metric in [UnitEuclideanMetric(), DiagEuclideanMetric(vec(var(temp; dims=2))), DenseEuclideanMetric(cov(temp'))]
        h = Hamiltonian(metric, _logπ, _dlogπdθ)
        ϵ = 0.02
        n_steps = 20
        p = TakeLastProposal(StaticTrajectory(Leapfrog(ϵ), n_steps))

        θ_init = randn(D)
        n_samples = 50_000
        samples = HMC.sample(h, p, θ_init, n_samples)

        @test mean(samples) ≈ zeros(D) atol=RNDATOL
    end
end
