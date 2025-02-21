using AdvancedHMC
using ReTest
import Aqua

@testset "Aqua" begin
    Aqua.test_all(AdvancedHMC)
end
