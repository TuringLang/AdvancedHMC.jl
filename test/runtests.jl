using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "Transition/trajectory",
        "Adaptation/precond",
        "hamiltonian",
        "integrator",
        "sampler",
        "models",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
