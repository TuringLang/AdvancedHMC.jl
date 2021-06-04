using Distributed, Test, CUDA

println("Environment variables for testing")
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

    if CUDA.functional()
        @eval module TestCUDA
            include("cuda.jl")
        end
    else
        @warn "Skipping GPU tests because no GPU available."
    end

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
