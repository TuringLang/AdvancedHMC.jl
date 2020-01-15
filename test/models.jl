using Test, Random, AdvancedHMC, ForwardDiff
using Statistics: mean
include("common.jl")
include(joinpath(splitpath(@__DIR__)[1:end-1]..., "benchmarks", "targets", "gdemo.jl"))

@testset "models" begin

    @testset "gdemo" begin
        res = run_nuts(2, ℓπ_gdemo; rng=MersenneTwister(1), verbose=true, drop_warmup=true)

        θ̂ = mean(map(invlink_gdemo, res.samples))

        @test θ̂ ≈ [49 / 24, 7 / 6] atol=RNDATOL
    end

end # @testset