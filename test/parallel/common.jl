# Common setup for parallel tests
#
# This file provides a unified way to access the Parallel module:
# - In CI: Uses the installed AdvancedHMC.Parallel module (for coverage tracking)
# - Standalone: Includes the module directly (for quick local testing)

using Test
using LinearAlgebra
using Random
using Statistics

# Check if AdvancedHMC is available and has the Parallel module
const USE_INSTALLED_PACKAGE = try
    @eval using AdvancedHMC
    isdefined(AdvancedHMC, :Parallel)
catch
    false
end

if USE_INSTALLED_PACKAGE
    # Use the installed package's Parallel module
    using AdvancedHMC.Parallel
else
    # Include the module directly for standalone testing
    include(joinpath(@__DIR__, "../../src/parallel/Parallel.jl"))
    using .Parallel
end
