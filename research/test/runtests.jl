using Comonicon, ReTest

# using Pkg
# Pkg.develop(path = dirname(dirname(@__DIR__)))
# Pkg.add(url = "https://github.com/xukai92/VecTargets.jl.git")

# include the sampler utility
include("../src/sampler_utility.jl")

# include the tests for experimental HMC
include("relativistic.jl")
include("riemannian.jl")

@main function runtests(patterns...; dry::Bool = false)
    retest(patterns...; dry = dry, verbose = Inf)
end
