using Distributed, Test

@testset "AdvancedHMC" begin
    tests = [
        "Kernel/trajectory",
        "Adaptation/precond",
        "hamiltonian",
        "integrator",
        "models",
        "sampler",
        "sampler_mat",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
