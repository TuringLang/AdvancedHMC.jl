using Distributed, Test, CUDA

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

    if CUDA.functional()
        @eval module Test_cuda
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
