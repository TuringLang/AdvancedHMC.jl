using AdvancedHMC
using Test
using Aqua: Aqua
using JET
using ForwardDiff

Test.@testset "Aqua" begin
    Aqua.test_all(AdvancedHMC)
end

Test.@testset "JET" begin
    JET.test_package(AdvancedHMC; target_defined_modules=true)
end
