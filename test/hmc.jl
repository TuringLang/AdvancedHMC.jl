using Test, AdvancedHMC
using Statistics: mean, var, cov
include("common.jl")

@testset "HMC and NUTS" begin
    θ_init = randn(D)
    ϵ = 0.02
    n_steps = 20
    n_samples = 100_000

    temp = randn(D,100)
    for metric in [UnitEuclideanMetric(θ_init), DiagEuclideanMetric(θ_init, vec(var(temp; dims=2))), DenseEuclideanMetric(θ_init, cov(temp'))]
        h = Hamiltonian(metric, logπ, ∂logπ∂θ)
        for prop in [TakeLastProposal(Leapfrog(ϵ), n_steps), SliceNUTS(Leapfrog(find_good_eps(h, θ_init)))]
            @time samples = AdvancedHMC.sample(h, prop, θ_init, n_samples)
            @test mean(samples) ≈ zeros(D) atol=RNDATOL
        end
    end
end
