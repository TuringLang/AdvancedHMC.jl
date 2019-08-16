using Test, AdvancedHMC
using Random
include("common.jl")

ϵ = 0.01
lf = Leapfrog(ϵ)

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
τ = NUTS(Leapfrog(find_good_eps(h, θ_init)))
r_init = AdvancedHMC.rand(h.metric)

@testset "Passing random number generator" begin
    for seed in [1234, 5678, 90]
        rng = MersenneTwister(seed)
        z = AdvancedHMC.phasepoint(h, θ_init, r_init)
        z1′, _ = AdvancedHMC.transition(rng, τ, h, z)

        rng = MersenneTwister(seed)
        z = AdvancedHMC.phasepoint(h, θ_init, r_init)
        z2′, _ = AdvancedHMC.transition(rng, τ, h, z)

        @test z1′.θ == z2′.θ
        @test z1′.r == z2′.r
    end
end

@testset "TreeSampler" begin
    logu = rand()
    s1 = AdvancedHMC.SliceTreeSampler(logu, 1) 
    s2 = AdvancedHMC.SliceTreeSampler(logu, 2) 
    s3 = AdvancedHMC.combine(s1, s2)
    @test s3.logu == logu
    @test s3.n == 1 + 2

    w1 = rand()
    s1 = AdvancedHMC.MultinomialTreeSampler(log(w1))
    w2 = rand()
    s2 = AdvancedHMC.MultinomialTreeSampler(log(w2))
    s3 = AdvancedHMC.combine(s1, s2)
    @test s3.logw == log(w1 + w2)
end
