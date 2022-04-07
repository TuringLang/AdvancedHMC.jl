using Random
using Test
using LinearAlgebra

using Turing
using Turing.DynamicPPL

using StatsFuns: logistic

@testset "Turing" begin
    Turing.setprogress!(false)

    Random.seed!(100)

    # Load test utilities.
    testdir(args...) = joinpath(pathof(Turing), "..", "..", "test", args...)
    include(testdir("test_utils", "AllUtils.jl"))

    # Test HMC.
    include(testdir("inference", "hmc.jl"))
end
