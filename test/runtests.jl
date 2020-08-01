using Distributed, Test

println("Envronment variables for testing")
println(ENV)

@testset "AdvancedHMC" begin
    tests = [
        "metric",
        "hamiltonian",
        "integrator",
        "trajectory",
        "adaptation",
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
