using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "metric",
        "hamiltonian",
        "integrator",
        "trajectory",
        "adaptation",
        "regression",
        "sampler",
        "sampler-vec",
        "demo",
        "models",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
