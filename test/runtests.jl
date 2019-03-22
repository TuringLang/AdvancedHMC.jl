using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "hamiltonian",
        "integrator",
        "proposal",
        "hmc",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
