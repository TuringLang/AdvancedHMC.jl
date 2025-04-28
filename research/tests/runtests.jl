using Comonicon, ReTest

using Pkg;
Pkg.add(; url="https://github.com/chalk-lab/MCMCLogDensityProblems.jl.git");

# include the source code for experimental HMC
include("../src/relativistic_hmc.jl")
include("../src/riemannian_hmc.jl")

# include the tests for experimental HMC
include("relativistic_hmc.jl")
include("riemannian_hmc.jl")

Comonicon.@main function runtests(patterns...; dry::Bool=false)
    return retest(patterns...; dry=dry, verbose=Inf)
end
