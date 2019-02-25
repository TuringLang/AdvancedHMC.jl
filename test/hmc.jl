using Test, HMC
using Statistics: mean
include("common.jl")

@testset "HMC" begin
    h = Hamiltonian(UnitEuclideanMetric(), _logπ, _dlogπdθ)
    ϵ = 0.02
    n_steps = 20
    p = TakeLastProposal(StaticTrajectory(Leapfrog(ϵ), n_steps))

    θ_init = randn(D)
    n_samples = 50_000
    samples = HMC.sample(h, p, θ_init, n_samples)

    @test mean(samples) ≈ zeros(D) atol=RNDATOL
end
