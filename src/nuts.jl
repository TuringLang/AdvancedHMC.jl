module NUTSJulia

export naive_NUTS, eff_NUTS, NUTS

using ForwardDiff

# Constant used in the base case of `build_tree`
# 1000 is the recommended value from Hoffman et al. (2011)
const Δ_max = 1000

include("naive_nuts.jl")
include("eff_nuts.jl")
include("nuts_with_dual_averaging.jl")

end
