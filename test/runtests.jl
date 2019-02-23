using Test

@testset "integrator.jl" begin
    include("integrator.jl")
end

# Sampler
include("hmc.jl")
