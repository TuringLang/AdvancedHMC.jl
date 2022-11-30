using ReTest, AdvancedHMC
using AdvancedHMC
using AdvancedHMC: RelativisticKinetic
using LinearAlgebra: dot

@testset "Hamiltonian" begin
    f = x -> dot(x, x)
    g = x -> 2x
    metric = UnitEuclideanMetric(10)
    h = Hamiltonian(metric, RelativisticKinetic(1.0, 1.0), f, g)
    @test h.kinetic isa RelativisticKinetic
end