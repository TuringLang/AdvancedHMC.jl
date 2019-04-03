using Test, AdvancedHMC
using Random
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
prop = NUTS(Leapfrog(find_good_eps(h, θ_init)))
r_init = AdvancedHMC.rand_momentum(h)

@testset "Passing random number generator" begin
    for seed in [1234, 5678, 90]
        rng = MersenneTwister(seed)
        θ1, r1 = AdvancedHMC.transition(rng, prop, h, θ_init, r_init)

        rng = MersenneTwister(seed)
        θ2, r2 = AdvancedHMC.transition(rng, prop, h, θ_init, r_init)

        @test θ1 == θ2
        @test r1 == r2
    end
end
