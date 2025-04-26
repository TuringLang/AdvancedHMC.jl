using AdvancedHMC
using ReTest
using Aqua: Aqua
using JET
using ForwardDiff

@testset "Aqua" begin
    Aqua.test_all(AdvancedHMC)
end

@testset "JET" begin
    JET.test_package(AdvancedHMC; target_defined_modules=true)
end
