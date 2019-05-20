using Test, AdvancedHMC
import AdvancedHMC: DualValue, PhasePoint

include("common.jl")

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
r_init = ones(D)

@test -AdvancedHMC.neg_energy(h, r_init, θ_init) == D / 2


init_z1() = PhasePoint([NaN], [NaN], DualValue(0.,[0.]), DualValue(0.,[0.]))
init_z2() = PhasePoint([Inf], [Inf], DualValue(0.,[0.]), DualValue(0.,[0.]))

@test_logs (:warn, "The current proposal will be rejected (due to numerical error(s))...") init_z1()
@test_logs (:warn, "The current proposal will be rejected (due to numerical error(s))...") init_z2()

z1 = init_z1()
z2 = init_z2()

@test z1.ℓπ.value == z1.ℓπ.value
@test z1.ℓκ.value == z1.ℓκ.value
