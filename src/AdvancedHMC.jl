module AdvancedHMC

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using LinearAlgebra: cholesky
using Statistics: mean, var
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot
using LazyArrays: BroadcastArray
using Random: GLOBAL_RNG, AbstractRNG

# Notations
# d - dimension of sampling sapce
# π(θ) - target distribution
# r - momentum variable

include("metric.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog
include("proposal.jl")
export TakeLastProposal, SliceNUTS

include("adaptation/Adaptation.jl")
using .Adaptation
export find_good_eps, DualAveraging
include("diagnosis.jl")
include("sampler.jl")

end # module
