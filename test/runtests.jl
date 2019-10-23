using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "adaptation/precond",
        "hamiltonian",
        "integrator",
        "trajectory",
        "models",
        "hmc",
        "hmc_matpara",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
