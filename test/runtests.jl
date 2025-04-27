using Comonicon
using FillArrays
using AdvancedHMC: AdvancedHMC
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD
using MCMCChains
using OrdinaryDiffEq
using ReTest

println("Environment variables for testing")
println(ENV)

const DIRECTORY_AdvancedHMC = dirname(dirname(pathof(AdvancedHMC)))
const DIRECTORY_Turing_tests = joinpath(DIRECTORY_AdvancedHMC, "test", "turing")
const GROUP = get(ENV, "AHMC_TEST_GROUP", "AdvancedHMC")

include("common.jl")

if GROUP == "All" || GROUP == "AdvancedHMC"
    using ReTest

    include("quality.jl")
    include("metric.jl")
    include("hamiltonian.jl")
    include("integrator.jl")
    include("trajectory.jl")
    include("adaptation.jl")
    include("sampler.jl")
    include("sampler-vec.jl")
    include("demo.jl")
    include("models.jl")
    include("abstractmcmc.jl")
    include("mcmcchains.jl")
    include("constructors.jl")

    Comonicon.@main function runtests(patterns...; dry::Bool=false)
        return retest(patterns...; dry=dry, verbose=Inf)
    end
end

if GROUP == "All" || GROUP == "Experimental"
    using Pkg
    # activate separate test environment
    Pkg.activate(joinpath(DIRECTORY_AdvancedHMC, "research"))
    Pkg.develop(PackageSpec(; path=DIRECTORY_AdvancedHMC))
    Pkg.instantiate()
    include(joinpath(DIRECTORY_AdvancedHMC, "research/tests", "runtests.jl"))
end

if GROUP == "All" || GROUP == "Downstream"
    using Pkg
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
        include(joinpath("turing", "runtests.jl"))
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        # If we can't resolve that means this is incompatible by SemVer and this is fine
        # It means we marked this as a breaking change, so we don't need to worry about
        # Mistakenly introducing a breaking change, as we have intentionally made one
        @info "Not compatible with this release. No problem." exception = err
    end
end
