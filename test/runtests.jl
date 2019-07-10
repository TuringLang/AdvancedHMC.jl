using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "adaptation/precond",
        "hamiltonian",
        "integrator",
        "trajectory",
        "hmc",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
