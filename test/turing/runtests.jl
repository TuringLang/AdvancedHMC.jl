using Random
using Test
using LinearAlgebra

using Turing
using Turing.DynamicPPL

using LogExpFunctions: logistic

Test.@testset "Turing" begin
    Turing.setprogress!(false)

    Random.seed!(100)

    # Load test utilities. Turing's test/ layout isn't a stable interface, so this
    # is pinned to the specific files hmc.jl needs as of Turing 0.46.
    testdir(args...) = joinpath(pathof(Turing), "..", "..", "test", args...)
    include(testdir("test_utils", "models.jl"))
    include(testdir("test_utils", "numerical_tests.jl"))

    # Test HMC.
    include(testdir("mcmc", "hmc.jl"))
end
