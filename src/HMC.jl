module HMC

using LinearAlgebra: cholesky
using Statistics: mean, var

# Notations
# d - dimension of sampling sapce
# π(θ) - target distribution
# r - momentum variable
# V - potential energy
# K - kinetic energy
# H - Hamiltonian energy

include("metric.jl")
export UnitMetric, DiagMetric, DenseMetric
include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog
include("trajectory.jl")
export LastFromTraj
export StaticTrajectory
include("diagnosis.jl")
include("sampler.jl")

end # module
