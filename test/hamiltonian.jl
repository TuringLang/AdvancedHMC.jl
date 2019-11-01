using Test, AdvancedHMC
using AdvancedHMC: DualValue, PhasePoint
using LinearAlgebra: dot, diagm


include("common.jl")

@testset "PhasePoint" begin
    init_z1() = PhasePoint([NaN], [NaN], DualValue(0.,[0.]), DualValue(0.,[0.]))
    init_z2() = PhasePoint([Inf], [Inf], DualValue(0.,[0.]), DualValue(0.,[0.]))

    @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z1()
    @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z2()

    z1 = init_z1()
    z2 = init_z2()

    @test z1.ℓπ.value == z1.ℓπ.value
    @test z1.ℓκ.value == z1.ℓκ.value
end

@testset "Metric" begin
    n_tests = 10

    for _ in 1:n_tests
        θ_init = randn(D)
        r_init = randn(D)

        h = Hamiltonian(UnitEuclideanMetric(D), ℓπ, ∂ℓπ∂θ)
        @test -AdvancedHMC.neg_energy(h, r_init, θ_init) == sum(abs2, r_init) / 2
        @test AdvancedHMC.∂H∂r(h, r_init) == r_init

        M⁻¹ = ones(D) + abs.(randn(D))
        h = Hamiltonian(DiagEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
        @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * diagm(0 => M⁻¹) * r_init / 2
        @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ .* r_init

        m = randn(D, D)
        M⁻¹ = m' * m
        h = Hamiltonian(DenseEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
        @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * M⁻¹ * r_init / 2
        @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ * r_init
    end
end