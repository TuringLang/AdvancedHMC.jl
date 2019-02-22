module HMC

using LinearAlgebra: cholesky

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
include("sampler.jl")
export sample
include("diagnosis.jl")

end # module
