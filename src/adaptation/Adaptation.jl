module Adaptation

using Statistics: middle
using ..AdvancedHMC: DEBUG, GLOBAL_RNG, AbstractRNG, Hamiltonian, rand_momentum, hamiltonian_energy, Leapfrog, step, AbstractProposal

include("stepsize.jl")

export find_good_eps, AbstractAdapter, DualAveraging, adapt!, update

end # module
