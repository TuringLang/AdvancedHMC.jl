using AdvancedHMC
using ReTest
using Aqua: Aqua

@testset "Aqua" begin
    Aqua.test_all(AdvancedHMC)
end
