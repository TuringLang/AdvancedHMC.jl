using Test

@testset "hamiltonian.jl" begin
    include("hamiltonian.jl")
end

@testset "integrator.jl" begin
    include("integrator.jl")
end

@testset "proposal.jl" begin
    include("proposal.jl")
end

@testset "Integrated tests" begin
    include("hmc.jl")
end
