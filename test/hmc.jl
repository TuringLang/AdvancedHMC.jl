using Test, HMC
using Statistics: mean
include("common.jl")

@testset "HMC" begin
    h = Hamiltonian(UnitMetric(), _logπ, _dlogπdθ)
    ϵ = 0.01
    n_steps = 10
    p = TakeLastProposal(StaticTrajectory(Leapfrog(ϵ), n_steps))

    θ_init = randn(D)
    n_samples = 10_000
    samples = HMC.sample(h, p, θ_init, n_samples)

    @test mean(samples) ≈ zeros(D) atol=RNDATOL
end
