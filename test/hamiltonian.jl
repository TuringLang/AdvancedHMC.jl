using Test, AdvancedHMC

include("common.jl")

θ_init = randn(D)
h = Hamiltonian(UnitEuclideanMetric(D), logπ, ∂logπ∂θ)
r_init = ones(D)

@test AdvancedHMC.kinetic_energy(h, r_init, θ_init) == D / 2
