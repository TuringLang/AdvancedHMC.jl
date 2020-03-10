using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "adaptation/precond",
        "trajectory",
        "hamiltonian",
        "integrator",
        "demo",
        "models",
        "sampler",
        "sampler-vec",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
