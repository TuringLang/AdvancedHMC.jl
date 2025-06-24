module AdvancedHMCResearchTests

using ReTest

# include the source code for experimental HMC
include("../src/relativistic_hmc.jl")

# include the tests for experimental HMC
include("relativistic_hmc.jl")

end
