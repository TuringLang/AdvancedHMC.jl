using Test, AdvancedHMC, Random
using Statistics: mean
using LinearAlgebra: dot
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
    n_samples = 10_000
    z1 = AdvancedHMC.phasepoint(h, zeros(D), r_init)
    z2 = AdvancedHMC.phasepoint(h, ones(D), r_init)

    rng = MersenneTwister(1234)

    logu = rand()
    n1 = 2
    s1 = AdvancedHMC.SliceTreeSampler(z1, logu, n1) 
    n2 = 1
    s2 = AdvancedHMC.SliceTreeSampler(z2, logu, n2) 
    s3 = AdvancedHMC.combine(rng, s1, s2)
    @test s3.logu == logu
    @test s3.n == n1 + n2

    
    s3_θ = Vector(undef, n_samples)
    for i = 1:n_samples
        s3_θ[i] = AdvancedHMC.combine(rng, s1, s2).zcand.θ
    end
    @test mean(s3_θ) ≈ ones(D) * n2 / (n1 + n2) rtol=0.01

    w1 = 100
    s1 = AdvancedHMC.MultinomialTreeSampler(z1, log(w1))
    w2 = 150
    s2 = AdvancedHMC.MultinomialTreeSampler(z2, log(w2))
    s3 = AdvancedHMC.combine(rng, s1, s2)
    @test s3.logw ≈ log(w1 + w2)

    s3_θ = Vector(undef, n_samples)
    for i = 1:n_samples
        s3_θ[i] = AdvancedHMC.combine(rng, s1, s2).zcand.θ
    end
    @test mean(s3_θ) ≈ ones(D) * w2 / (w1 + w2) rtol=0.01
end

@testset "TerminationCriterion" begin
    z1 = AdvancedHMC.phasepoint(h, θ_init, randn(D))
    c1 = AdvancedHMC.NoUTurn(z1)
    z2 = AdvancedHMC.phasepoint(h, θ_init, randn(D))
    c2 = AdvancedHMC.NoUTurn(z2)
    c3 = AdvancedHMC.combine(c1, c2)
    @test c1 == c2 == c3

    r1 = randn(D)
    z1 = AdvancedHMC.phasepoint(h, θ_init, r1)
    c1 = AdvancedHMC.GeneralisedNoUTurn(z1) 
    r2 = randn(D)
    z2 = AdvancedHMC.phasepoint(h, θ_init, r2)
    c2 = AdvancedHMC.GeneralisedNoUTurn(z2) 
    c3 = AdvancedHMC.combine(c1, c2)
    @test c3.rho == r1 + r2
end

@testset "Termination" begin
    t00 = AdvancedHMC.Termination(false, false)
    t01 = AdvancedHMC.Termination(false, true)
    t10 = AdvancedHMC.Termination(true, false)
    t11 = AdvancedHMC.Termination(true, true)

    @test AdvancedHMC.isterminated(t00) == false
    @test AdvancedHMC.isterminated(t01) == true
    @test AdvancedHMC.isterminated(t10) == true
    @test AdvancedHMC.isterminated(t11) == true

    @test AdvancedHMC.isterminated(t00 * t00) == false
    @test AdvancedHMC.isterminated(t00 * t01) == true
    @test AdvancedHMC.isterminated(t00 * t10) == true
    @test AdvancedHMC.isterminated(t00 * t11) == true

    @test AdvancedHMC.isterminated(t01 * t00) == true
    @test AdvancedHMC.isterminated(t01 * t01) == true
    @test AdvancedHMC.isterminated(t01 * t10) == true
    @test AdvancedHMC.isterminated(t01 * t11) == true

    @test AdvancedHMC.isterminated(t10 * t00) == true
    @test AdvancedHMC.isterminated(t10 * t01) == true
    @test AdvancedHMC.isterminated(t10 * t10) == true
    @test AdvancedHMC.isterminated(t10 * t11) == true

    @test AdvancedHMC.isterminated(t11 * t00) == true
    @test AdvancedHMC.isterminated(t11 * t01) == true
    @test AdvancedHMC.isterminated(t11 * t10) == true
    @test AdvancedHMC.isterminated(t11 * t11) == true
end

@testset "FullBinaryTree" begin
    @warn "FullBinaryTree not tested"
end