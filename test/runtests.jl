using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "Adaptation/precond",
        "hamiltonian",
        "integrator",
        "trajectory",
        "models",
        "hmc",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
