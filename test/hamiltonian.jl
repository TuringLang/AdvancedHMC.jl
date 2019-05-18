using Test, AdvancedHMC
import AdvancedHMC: DualValue, PhasePoint

include("common.jl")

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
r_init = ones(D)

@test AdvancedHMC.kinetic_energy(h, r_init, θ_init) == D / 2

z1 = PhasePoint([NaN], [NaN], DualValue(0.,[0.]), DualValue(0.,[0.]))
z2 = PhasePoint([Inf], [Inf], DualValue(0.,[0.]), DualValue(0.,[0.]))

@test z1.logπ.value == z1.logπ.value
@test z1.logκ.value == z1.logκ.value
