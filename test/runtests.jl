using Distributed, Test, CUDA, Pkg

using AdvancedHMC: AdvancedHMC

println("Environment variables for testing")
println(ENV)

const DIRECTORY_AdvancedHMC = dirname(dirname(pathof(AdvancedHMC)))
const DIRECTORY_Turing_tests = joinpath(DIRECTORY_AdvancedHMC, "test", "turing")
const GROUP = get(ENV, "AHMC_TEST_GROUP", "All")

@testset "AdvancedHMC" begin
    if GROUP == "All" || GROUP == "AdvancedHMC"
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
            "abstractmcmc"
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

    if GROUP == "All" || GROUP == "Downstream"
        @testset "turing" begin
            try
                # activate separate test environment
                Pkg.activate(DIRECTORY_Turing_tests)
                Pkg.develop(PackageSpec(; path=DIRECTORY_AdvancedHMC))
                Pkg.instantiate()

                # make sure that the new environment is considered `using` and `import` statements
                # (not added automatically on Julia 1.3, see e.g. PR #209)
                if !(joinpath(DIRECTORY_Turing_tests, "Project.toml") in Base.load_path())
                    pushfirst!(LOAD_PATH, DIRECTORY_Turing_tests)
                end

                # Avoids conflicting namespaces, e.g. `NUTS` used in Turing.jl's tests
                # refers to `Turing.NUTS` not `AdvancedHMC.NUTS`.
                @eval module TuringIntegrationTests
                include(joinpath("turing", "runtests.jl"))
                end
            catch err
                err isa Pkg.Resolve.ResolverError || rethrow()
                # If we can't resolve that means this is incompatible by SemVer and this is fine
                # It means we marked this as a breaking change, so we don't need to worry about
                # Mistakenly introducing a breaking change, as we have intentionally made one
                @info "Not compatible with this release. No problem." exception = err
            end
        end
    end
end
