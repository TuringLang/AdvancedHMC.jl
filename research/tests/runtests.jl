using ReTest

# include the source code for relativistic HMC
include("../src/relativistic_hmc.jl")

# include the tests for relativistic HMC
include("relativistic_hmc.jl")

@main function runtests(patterns...; dry::Bool=false)
    retest(patterns...; dry=dry, verbose=Inf)
end