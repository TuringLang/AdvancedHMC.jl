using ReTest, AdvancedHMC

include("../src/riemannian_hmc-sampler.jl")

using LinearAlgebra: dot

@testset "Relativistic" begin

@testset "Hamiltonian" begin
    @testset "Construction" begin
        f = x -> dot(x, x)
        g = x -> 2x
        metric = UnitEuclideanMetric(10)
        h = Hamiltonian(metric, RelativisticKinetic(1.0, 1.0), f, g)
        @test h.kinetic isa RelativisticKinetic
    end
end

end