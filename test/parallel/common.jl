# Common setup for parallel tests
#
# This file provides a unified way to access the Parallel module:
# - In CI: Uses the installed AdvancedHMC.Parallel module (for coverage tracking)
# - Standalone: Includes the module directly (for quick local testing)

# Guard against multiple includes
if !@isdefined(_PARALLEL_COMMON_LOADED)

    # Only import Test if ReTest is not already loaded (avoids @testset conflict)
    if !isdefined(Main, :ReTest)
        using Test
    end
    using LinearAlgebra
    using Random
    using Statistics

    # Check if AdvancedHMC is available and has the Parallel module
    _use_installed = try
        @eval using AdvancedHMC
        isdefined(AdvancedHMC, :Parallel)
    catch
        false
    end

    if _use_installed
        using AdvancedHMC.Parallel
    else
        include(joinpath(@__DIR__, "../../src/parallel/Parallel.jl"))
        using .Parallel
    end

    const _PARALLEL_COMMON_LOADED = true
end # guard
