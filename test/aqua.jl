using AdvancedHMC
using Test
import Aqua

@testset "Aqua" begin
    Aqua.test_all(AdvancedHMC)
end
