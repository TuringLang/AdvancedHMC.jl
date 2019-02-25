using Test

@testset "integrator.jl" begin
    include("integrator.jl")
end

@testset "Integrated tests" begin
    include("hmc.jl")
end
