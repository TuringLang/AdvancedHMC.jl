using Comonicon, ReTest

# include the source code for experimental HMC
include("../src/relativistic_hmc.jl")
include("../src/riemannian_hmc.jl")

# include the tests for experimental HMC
include("relativistic_hmc.jl")
include("riemannian_hmc.jl")

@main function runtests(patterns...; dry::Bool = false)
    retest(patterns...; dry = dry, verbose = Inf)
end
