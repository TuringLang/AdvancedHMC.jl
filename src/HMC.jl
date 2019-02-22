module HMC

using LinearAlgebra: cholesky

# Notations
# d - dimension of sampling sapce
# π(θ) - target distribution
# r - momentum variable
# V - potential energy
# K - kinetic energy
# H - Hamiltonian energy

include("point.jl")
include("metric.jl")
include("hamiltonian.jl")
include("integrator.jl")
include("trajectory.jl")
include("sampler.jl")
include("diagnosis.jl")

end # module
