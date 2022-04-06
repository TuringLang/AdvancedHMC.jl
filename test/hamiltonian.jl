using ReTest, AdvancedHMC
using AdvancedHMC: DualValue, PhasePoint
using LinearAlgebra: dot, diagm


include("common.jl")

@testset "PhasePoint" begin
    for T in [Float32, Float64]
        init_z1() = PhasePoint(
            [T(NaN)],
            [T(NaN)],
            DualValue(zero(T), [zero(T)]),
            DualValue(zero(T), [zero(T)])
        )
        init_z2() = PhasePoint(
            [T(Inf)],
            [T(Inf)],
            DualValue(zero(T), [zero(T)]),
            DualValue(zero(T), [zero(T)])
        )

        @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z1()
        @test_logs (:warn, "The current proposal will be rejected due to numerical error(s).") init_z2()

        z1 = init_z1()
        z2 = init_z2()

        @test z1.ℓπ.value == z2.ℓπ.value
        @test z1.ℓκ.value == z2.ℓκ.value
    end
end

@testset "Metric" begin
    n_tests = 10

    for T in [Float32, Float64]
        for _ in 1:n_tests
            θ_init = randn(T, D)
            r_init = randn(T, D)

            h = Hamiltonian(UnitEuclideanMetric(T, D), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) == sum(abs2, r_init) / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == r_init

            M⁻¹ = ones(T, D) + abs.(randn(T, D))
            h = Hamiltonian(DiagEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * diagm(0 => M⁻¹) * r_init / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ .* r_init

            m = randn(T, D, D)
            M⁻¹ = m' * m
            h = Hamiltonian(DenseEuclideanMetric(M⁻¹), ℓπ, ∂ℓπ∂θ)
            @test -AdvancedHMC.neg_energy(h, r_init, θ_init) ≈ r_init' * M⁻¹ * r_init / 2
            @test AdvancedHMC.∂H∂r(h, r_init) == M⁻¹ * r_init
        end
    end
end
