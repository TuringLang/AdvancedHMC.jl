module HMC

using LinearAlgebra: cholesky
using Statistics: mean, var
using LinearAlgebra: Symmetric
using LazyArrays: BroadcastArray

# Notations
# d - dimension of sampling sapce
# π(θ) - target distribution
# r - momentum variable
# V - potential energy
# K - kinetic energy
# H - Hamiltonian energy

include("metric.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog
include("trajectory.jl")
export StaticTrajectory
include("proposal.jl")
export TakeLastProposal
include("diagnosis.jl")
include("sampler.jl")

end # module
